"""Database service for Appwrite integration."""

from appwrite.client import Client
from appwrite.services.databases import Databases
from appwrite.exception import AppwriteException
from typing import Optional, List, Dict
from loguru import logger
import asyncio
from functools import wraps

from app.models import UserContext, Message
from app.utils import retry_with_backoff


def async_appwrite(func):
    """
    Decorator to run synchronous Appwrite SDK calls in executor.
    
    The Appwrite Python SDK is synchronous, so we need to run it
    in a thread pool executor to avoid blocking the event loop.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
    return wrapper


class DatabaseService:
    """
    Service for managing user data in Appwrite database.
    
    Provides async methods for CRUD operations on user context data,
    including chat history, summaries, and user preferences.
    
    Requirements: 2.5, 3.4, 4.2, 13.5, 11.1
    """
    
    def __init__(
        self,
        endpoint: str,
        project_id: str,
        api_key: str,
        database_id: str,
        collection_id: str
    ):
        """
        Initialize Appwrite database service.
        
        Args:
            endpoint: Appwrite API endpoint URL
            project_id: Appwrite project ID
            api_key: Appwrite API key for authentication
            database_id: Database ID in Appwrite
            collection_id: Collection ID for user data
        """
        self.client = Client()
        self.client.set_endpoint(endpoint)
        self.client.set_project(project_id)
        self.client.set_key(api_key)
        
        self.databases = Databases(self.client)
        self.database_id = database_id
        self.collection_id = collection_id
        
        logger.info(
            f"DatabaseService initialized: endpoint={endpoint}, "
            f"project={project_id}, database={database_id}, collection={collection_id}"
        )

    async def get_user_context(self, user_id: str) -> Optional[UserContext]:
        """
        Fetch user context from Appwrite database.
        
        Retrieves user data including chat history, interests, summary,
        birthdate, and topics. Includes retry logic with exponential backoff.
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            UserContext object if user exists, None if not found
            
        Raises:
            AppwriteException: If database operation fails after retries
            
        Requirements: 2.3, 4.2, 13.5, 11.1
        """
        logger.info(f"Fetching user context for user_id={user_id} from Appwrite database")
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Wrap synchronous Appwrite call with retry logic
            # Capture variables to avoid closure issues
            db_id = self.database_id
            coll_id = self.collection_id
            doc_id = user_id
            
            async def _fetch():
                logger.debug(
                    f"Calling Appwrite get_document: database_id={db_id}, "
                    f"collection_id={coll_id}, document_id={doc_id}"
                )
                loop = asyncio.get_event_loop()
                
                def _sync_call():
                    return self.databases.get_document(db_id, coll_id, doc_id)
                
                return await loop.run_in_executor(None, _sync_call)
            
            document = await retry_with_backoff(_fetch, operation_name="get_user_context")
            
            duration = asyncio.get_event_loop().time() - start_time
            logger.info(
                f"User context retrieved for user_id={user_id} in {duration:.3f}s"
            )
            
            # Parse document into UserContext
            import json
            chat_history = []
            if "chatHistory" in document and document["chatHistory"]:
                for msg_str in document["chatHistory"]:
                    # Parse JSON string back to dict
                    msg = json.loads(msg_str) if isinstance(msg_str, str) else msg_str
                    chat_history.append(Message(
                        role=msg.get("role", "user"),
                        content=msg.get("content", "")
                    ))
            
            context = UserContext(
                chatHistory=chat_history,
                chatInterest=document.get("chatInterest"),
                userSummary=document.get("userSummary", ""),
                birthdate=document.get("birthdate"),
                topics=document.get("topics", [])
            )
            
            logger.debug(f"=== FETCHED USER CONTEXT (user_id={user_id}) ===")
            logger.debug(f"Chat Interest: {context.chatInterest}")
            logger.debug(f"Topics: {context.topics}")
            logger.debug(f"Birthdate: {context.birthdate}")
            logger.debug(f"User Summary: {context.userSummary if context.userSummary else '(empty)'}")
            logger.debug(f"Chat History ({len(context.chatHistory)} messages):")
            for idx, msg in enumerate(context.chatHistory):
                logger.debug(f"  [{idx}] {msg.role}: {msg.content}")
            logger.debug(f"=== END FETCHED USER CONTEXT ===")
            
            return context
            
        except AppwriteException as e:
            duration = asyncio.get_event_loop().time() - start_time
            
            # 404 means user doesn't exist - this is expected for new users
            if e.code == 404:
                logger.info(
                    f"User not found: user_id={user_id} (duration={duration:.3f}s)"
                )
                return None
            
            # Other errors are unexpected
            logger.error(
                f"Failed to fetch user context for user_id={user_id} "
                f"after {duration:.3f}s: {e.message} (code={e.code})"
            )
            raise
        except Exception as e:
            duration = asyncio.get_event_loop().time() - start_time
            logger.error(
                f"Unexpected error fetching user context for user_id={user_id} "
                f"after {duration:.3f}s: {e}",
                exc_info=True
            )
            raise

    async def update_chat_history(
        self,
        user_id: str,
        chat_history: List[Dict],
        user_summary: str = "",
        chat_interest: Optional[str] = None
    ) -> None:
        """
        Update user's chat history, summary, and interest in Appwrite.
        
        Updates the chatHistory, userSummary, and optionally chatInterest fields for an existing user.
        Includes retry logic with exponential backoff.
        
        Args:
            user_id: Unique user identifier
            chat_history: List of message dictionaries with 'role' and 'content'
            user_summary: Optional summary of older messages
            chat_interest: Optional user's interest topic (updated for first-time users)
            
        Raises:
            AppwriteException: If database operation fails after retries
            
        Requirements: 2.5, 3.4, 13.5, 11.1
        """
        logger.info(
            f"Updating chat history for user_id={user_id}: "
            f"{len(chat_history)} messages, summary_length={len(user_summary) if user_summary else 'No summary yet'}"
        )
        logger.debug(f"=== UPDATING CHAT HISTORY (user_id={user_id}) ===")
        logger.debug(f"User Summary: {user_summary if user_summary else '(empty)'}")
        logger.debug(f"Chat History ({len(chat_history)} messages):")
        import json
        for idx, msg in enumerate(chat_history):
            # msg is now a JSON string, parse it for logging
            msg_dict = json.loads(msg) if isinstance(msg, str) else msg
            logger.debug(f"  [{idx}] {msg_dict.get('role', 'unknown')}: {msg_dict.get('content', '')}")
        logger.debug(f"=== END UPDATING CHAT HISTORY ===")
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Wrap synchronous Appwrite call with retry logic
            # Capture variables to avoid closure issues
            db_id = self.database_id
            coll_id = self.collection_id
            doc_id = user_id
            update_data = {
                "chatHistory": chat_history,
                "userSummary": user_summary
            }
            
            # Only include chatInterest if it's provided
            if chat_interest is not None:
                update_data["chatInterest"] = chat_interest
            
            async def _update():
                logger.debug(
                    f"Calling Appwrite update_document: database_id={db_id}, "
                    f"collection_id={coll_id}, document_id={doc_id}"
                )
                loop = asyncio.get_event_loop()
                
                def _sync_call():
                    return self.databases.update_document(db_id, coll_id, doc_id, update_data)
                
                return await loop.run_in_executor(None, _sync_call)
            
            await retry_with_backoff(_update, operation_name="update_chat_history")
            
            duration = asyncio.get_event_loop().time() - start_time
            logger.info(
                f"Chat history updated for user_id={user_id} in {duration:.3f}s"
            )
            
        except AppwriteException as e:
            duration = asyncio.get_event_loop().time() - start_time
            logger.error(
                f"Failed to update chat history for user_id={user_id} "
                f"after {duration:.3f}s: {e.message} (code={e.code})"
            )
            raise
        except Exception as e:
            duration = asyncio.get_event_loop().time() - start_time
            logger.error(
                f"Unexpected error updating chat history for user_id={user_id} "
                f"after {duration:.3f}s: {e}",
                exc_info=True
            )
            raise

    async def create_user_context(
        self,
        user_id: str,
        context: UserContext
    ) -> None:
        """
        Create new user document in Appwrite.
        
        Creates a new user record with initial context data including
        chat history, interests, birthdate, and topics.
        Includes retry logic with exponential backoff.
        
        Args:
            user_id: Unique user identifier
            context: UserContext object with initial user data
            
        Raises:
            AppwriteException: If database operation fails after retries
            
        Requirements: 3.4, 13.5, 11.1
        """
        logger.info(
            f"Creating user context for user_id={user_id}: "
            f"chatInterest={context.chatInterest}, "
            f"topics={len(context.topics)}, "
            f"history={len(context.chatHistory)} messages"
        )
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Convert UserContext to JSON string format for Appwrite
            import json
            chat_history_data = [
                json.dumps({"role": msg.role, "content": msg.content})
                for msg in context.chatHistory
            ]
            
            data = {
                "chatHistory": chat_history_data,
                "chatInterest": context.chatInterest,
                "userSummary": context.userSummary,
                "birthdate": context.birthdate,
                "topics": context.topics
            }
            
            # Wrap synchronous Appwrite call with retry logic
            # Capture variables to avoid closure issues
            db_id = self.database_id
            coll_id = self.collection_id
            doc_id = user_id
            
            async def _create():
                logger.debug(
                    f"Calling Appwrite create_document: database_id={db_id}, "
                    f"collection_id={coll_id}, document_id={doc_id}"
                )
                loop = asyncio.get_event_loop()
                
                def _sync_call():
                    return self.databases.create_document(db_id, coll_id, doc_id, data)
                
                return await loop.run_in_executor(None, _sync_call)
            
            await retry_with_backoff(_create, operation_name="create_user_context")
            
            duration = asyncio.get_event_loop().time() - start_time
            logger.info(
                f"User context created for user_id={user_id} in {duration:.3f}s"
            )
            
        except AppwriteException as e:
            duration = asyncio.get_event_loop().time() - start_time
            logger.error(
                f"Failed to create user context for user_id={user_id} "
                f"after {duration:.3f}s: {e.message} (code={e.code})"
            )
            raise
        except Exception as e:
            duration = asyncio.get_event_loop().time() - start_time
            logger.error(
                f"Unexpected error creating user context for user_id={user_id} "
                f"after {duration:.3f}s: {e}",
                exc_info=True
            )
            raise
