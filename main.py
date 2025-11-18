"""
FastAPI Chat Agent Application.

A production-ready chat agent service with multi-provider LLM support,
intelligent caching, and web search capabilities.

Requirements: 1.1, 1.2, 1.3, 1.4, 10.3, 10.4, 10.5
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from loguru import logger
from contextlib import asynccontextmanager
import asyncio

from app.config import get_settings
from app.models import ChatRequest, ChatResponse, Message, UserContext
from app.cache import CacheManager
from app.db_service import DatabaseService
from app.ai_agent import AIAgent
from app.search import SearchService
from app.utils import configure_logging


# Global service instances
cache_manager: CacheManager = None
db_service: DatabaseService = None
ai_agent: AIAgent = None
search_service: SearchService = None
cleanup_task: asyncio.Task = None


async def periodic_cache_cleanup():
    """
    Background task to periodically clean up expired cache entries.
    
    Runs every hour to remove expired entries from DiskCache.
    DiskCache uses lazy deletion, so this ensures disk space is reclaimed.
    """
    while True:
        try:
            await asyncio.sleep(3600)  # Run every hour
            if cache_manager:
                count = await cache_manager.cleanup_expired()
                logger.info(f"Periodic cache cleanup completed: {count} entries removed")
        except asyncio.CancelledError:
            logger.info("Cache cleanup task cancelled")
            break
        except Exception as e:
            logger.error(f"Error in periodic cache cleanup: {e}", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    
    Handles initialization of services during startup and cleanup during shutdown.
    
    Requirements: 1.4
    """
    # Startup
    global cache_manager, db_service, ai_agent, search_service, cleanup_task
    
    settings = get_settings()
    
    # Configure logging
    configure_logging(
        log_level=settings.log_level,
        rotation=settings.log_rotation,
        retention=settings.log_retention
    )
    
    logger.info("Starting Chat Agent application...")
    
    # Initialize services
    cache_manager = CacheManager(
        cache_dir=settings.cache_directory,
        ttl=settings.cache_ttl_seconds
    )
    
    db_service = DatabaseService(
        endpoint=settings.appwrite_endpoint,
        project_id=settings.appwrite_project_id,
        api_key=settings.appwrite_api_key,
        database_id=settings.appwrite_database_id,
        collection_id=settings.appwrite_collection_id
    )
    
    search_service = SearchService(
        api_key=settings.brave_api_key,
        timeout=10.0
    )
    
    ai_agent = AIAgent(
        openai_key=settings.openai_api_key,
        gemini_key=settings.gemini_api_key,
        default_provider=settings.default_llm_provider,
        search_service=search_service
    )
    
    # Start background cache cleanup task
    cleanup_task = asyncio.create_task(periodic_cache_cleanup())
    logger.info("Background cache cleanup task started")
    
    logger.info("All services initialized successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Chat Agent application...")
    
    # Cancel background tasks
    if cleanup_task:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
    
    # Cleanup resources
    if cache_manager:
        await cache_manager.close()
    
    if search_service:
        await search_service.close()
    
    logger.info("Application shutdown complete")


# Initialize FastAPI app
app = FastAPI(
    title="Chat Agent API",
    version="1.0.0",
    description="Personalized AI chat agent with multi-provider LLM support",
    lifespan=lifespan
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns the health status of the application.
    
    Returns:
        dict: Health status information
        
    Requirements: 10.5
    """
    return {
        "status": "healthy",
        "service": "chat-agent",
        "version": "1.0.0"
    }


@app.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
async def chat_endpoint(request: Request, chat_request: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint for processing user messages.
    
    Handles both first-time and returning user interactions with intelligent
    context management, caching, and AI response generation.
    
    Args:
        request: FastAPI request object (for rate limiting)
        chat_request: Validated chat request payload
        
    Returns:
        ChatResponse: AI-generated response in Markdown format
        
    Raises:
        HTTPException: Various HTTP errors based on failure scenarios
        
    """
    user_id = chat_request.userId
    user_message = chat_request.userMessage
    is_first_time = chat_request.chatInterest
    interest_topic = chat_request.interestTopic
    
    logger.info(
        f"Chat request received: user_id={user_id}, "
        f"message_length={len(user_message)}, "
        f"is_first_time={is_first_time}"
    )
    
    try:
        settings = get_settings()
        
        # Step 1: Check cache for user context
        user_context = await cache_manager.get(user_id)
        
        # Step 2: Handle cache miss - fetch from database
        if user_context is None:
            logger.info(f"Cache miss for user_id={user_id}, fetching from database")
            user_context = await db_service.get_user_context(user_id)
            
            # If user doesn't exist in database either, create new context
            if user_context is None:
                logger.info(f"New user detected: user_id={user_id}")
                user_context = UserContext(
                    chatHistory=[],
                    chatInterest=interest_topic if is_first_time else None,
                    userSummary="",
                    birthdate=None,
                    topics=[]
                )
            else:
                # Cache the fetched context
                await cache_manager.set(user_id, user_context)
        
        # Step 3: Handle first-time user flow
        if is_first_time:
            logger.info(f"Processing first-time user: user_id={user_id}, topic={interest_topic}")
            
            # Use interestTopic as the initial message
            actual_message = interest_topic
            user_context.chatInterest = interest_topic
            
            # Build system prompt for first-time user
            system_prompt = ai_agent._build_system_prompt(
                user_context=user_context,
                is_first_message=True
            )
            
            # Prepare messages for AI (just the initial interest)
            messages = [
                {"role": "user", "content": actual_message}
            ]
            
        # Step 4: Handle returning user flow
        else:
            logger.info(f"Processing returning user: user_id={user_id}")
            
            actual_message = user_message
            
            # Build system prompt for returning user (includes summary if present)
            system_prompt = ai_agent._build_system_prompt(
                user_context=user_context,
                is_first_message=False
            )
            
            # Prepare messages with recent history
            max_context = settings.previous_message_context_length
            recent_messages = user_context.chatHistory[-max_context:] if user_context.chatHistory else []
            
            # Convert Message objects to dicts
            messages = [
                {"role": msg.role, "content": msg.content}
                for msg in recent_messages
            ]
            
            # Add current user message
            messages.append({"role": "user", "content": actual_message})
        
        # Step 5: Generate AI response
        logger.info(
            f"Generating AI response for user_id={user_id}\n"
            f"Provider: {settings.default_llm_provider}\n"
            f"Message count: {len(messages)}\n"
            f"System prompt length: {len(system_prompt)}\n"
            f"Is first time: {is_first_time}"
        )
        logger.debug(f"=== SYSTEM PROMPT ===\n{system_prompt}\n=== END SYSTEM PROMPT ===")
        logger.debug(f"=== MESSAGES TO SEND ===")
        for idx, msg in enumerate(messages):
            logger.debug(f"Message {idx} [{msg['role']}]: {msg.get('content', '')}")
        logger.debug(f"=== END MESSAGES ===")
        
        response_text, updated_messages = await ai_agent.generate_response(
            messages=messages,
            system_prompt=system_prompt,
            user_context=user_context,
            provider=settings.default_llm_provider
        )
        
        logger.debug(f"=== AI RESPONSE ===\n{response_text}\n=== END AI RESPONSE ===")
        logger.debug(f"Updated messages count: {len(updated_messages)}")
        
        # Step 6: Update chat history
        # Add user message to history
        user_context.chatHistory.append(
            Message(role="user", content=actual_message)
        )
        
        # Add assistant response to history
        user_context.chatHistory.append(
            Message(role="assistant", content=response_text)
        )
        
        logger.info(
            f"Chat history updated: user_id={user_id}, "
            f"total_messages={len(user_context.chatHistory)}"
        )
        logger.debug(f"=== FULL CHAT HISTORY (user_id={user_id}) ===")
        for idx, msg in enumerate(user_context.chatHistory):
            logger.debug(f"[{idx}] {msg.role}: {msg.content}")
        logger.debug(f"=== END CHAT HISTORY ===")
        
        # Step 7: Check if summarization is needed
        needs_summarization, _ = await cache_manager.check_and_summarize(
            user_id=user_id,
            context=user_context,
            max_messages=settings.previous_message_context_length,
            overlap=settings.overlap_count
        )
        
        if needs_summarization:
            logger.info(f"Triggering summarization for user_id={user_id}")
            
            # Calculate how many messages to summarize
            threshold = settings.previous_message_context_length + settings.overlap_count
            overflow_count = len(user_context.chatHistory) - threshold
            messages_to_summarize = user_context.chatHistory[:overflow_count]
            
            # Convert to dict format for summarization
            messages_dict = [
                {"role": msg.role, "content": msg.content}
                for msg in messages_to_summarize
            ]
            
            # Generate summary with previous summary if it exists
            summary = await ai_agent.summarize_messages(
                messages_dict,
                previous_summary=user_context.userSummary
            )

            # Update user summary with the compressed result
            user_context.userSummary = summary
            
            # Trim chat history to keep only recent messages
            user_context.chatHistory = user_context.chatHistory[-settings.previous_message_context_length:]
            
            logger.info(
                f"Summarization complete: user_id={user_id}, "
                f"summary_length={len(summary)}, "
                f"remaining_messages={len(user_context.chatHistory)}"
            )
        
        # Step 8: Update cache and database
        await cache_manager.set(user_id, user_context)
        
        # Convert chat history to JSON string format for database
        import json
        chat_history_dict = [
            json.dumps({"role": msg.role, "content": msg.content})
            for msg in user_context.chatHistory
        ]
        
        # Check if user exists in database
        existing_user = await db_service.get_user_context(user_id)
        
        if existing_user is None:
            # Create new user in database
            await db_service.create_user_context(user_id, user_context)
        else:
            # Update existing user
            await db_service.update_chat_history(
                user_id=user_id,
                chat_history=chat_history_dict,
                user_summary=user_context.userSummary,
                chat_interest=user_context.chatInterest
            )
        
        logger.info(f"Chat request completed successfully for user_id={user_id}")
        
        # Step 9: Return formatted response
        return ChatResponse(response=response_text)
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log error with comprehensive context
        import traceback
        tb = traceback.format_exc()
        
        logger.error(
            f"=== CHAT REQUEST FAILED ===\n"
            f"User ID: {user_id}\n"
            f"Message: {user_message[:100] if user_message else 'N/A'}...\n"
            f"Is first time: {is_first_time}\n"
            f"Interest topic: {interest_topic}\n"
            f"Error type: {type(e).__name__}\n"
            f"Error message: {str(e)}\n"
            f"Full traceback:\n{tb}\n"
            f"=========================="
        )
        
        # Return appropriate HTTP error
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {type(e).__name__}: {str(e)}"
        )
