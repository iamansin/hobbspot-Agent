"""AI Agent with multi-provider LLM support and function calling."""

import asyncio
from typing import List, Dict, Optional, Tuple, Any
from loguru import logger
from openai import AsyncOpenAI
import google.generativeai as genai

from app.models import UserContext, Message
from app.utils import retry_with_backoff


class AIAgent:
    """
    AI Agent with support for multiple LLM providers and function calling.
    
    Provides a unified interface for generating chat responses using either
    OpenAI or Google Gemini. Includes function calling support for web search,
    message summarization, and comprehensive logging.
    
    Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 5.2, 5.3, 13.6, 13.7, 13.11, 11.3
    """
    
    def __init__(
        self,
        openai_key: str,
        gemini_key: str,
        default_provider: str = "openai",
        search_service: Optional[Any] = None
    ):
        """
        Initialize AI Agent with LLM providers.
        
        Args:
            openai_key: OpenAI API key
            gemini_key: Google Gemini API key
            default_provider: Default LLM provider to use ("openai" or "gemini")
            search_service: Optional SearchService instance for function calling
        """
        # Initialize OpenAI client
        self.openai_client = AsyncOpenAI(api_key=openai_key)
        
        # Initialize Gemini client
        genai.configure(api_key=gemini_key)
        
        self.default_provider = default_provider
        self.search_service = search_service
        
        # Function calling schema
        self.function_schema = {
            "name": "web_search",
            "description": "Search the web for current information when the user asks about recent events, news, or information that may not be in your training data",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant information"
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of search results to return (default: 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
        
        logger.info(f"AIAgent initialized: default_provider={default_provider}")
    
    def _build_system_prompt(
        self,
        user_context: UserContext,
        is_first_message: bool
    ) -> str:
        """
        Build personalized system prompt based on user context.
        
        Creates a dynamic system prompt that incorporates user personalization
        data including interests, birthdate, topics, and conversation summary.
        
        Args:
            user_context: User context with personalization data
            is_first_message: Whether this is the user's first interaction
            
        Returns:
            Formatted system prompt string
            
        Requirements: 6.1
        """
        prompt_parts = [
            "You are a helpful and personalized AI assistant. Your goal is to provide "
            "relevant, engaging responses tailored to the user's interests and context."
        ]
        
        # Add user interest information
        if user_context.chatInterest:
            prompt_parts.append(
                f"\n###The user is interested in: {user_context.chatInterest}"
            )
        
        # Add topics
        if user_context.topics:
            topics_str = ", ".join(user_context.topics)
            prompt_parts.append(
                f"\n###The user's topics of interest include: {topics_str}"
            )
        
        # Add birthdate if available
        if user_context.birthdate:
            prompt_parts.append(
                f"\n###User's birthdate: {user_context.birthdate}"
            )
        
        # Add conversation summary for returning users
        if not is_first_message and user_context.userSummary:
            prompt_parts.append(
                f"\n\n**Previous conversation summary:**\n{user_context.userSummary}"
            )
        
        # Add instructions
        prompt_parts.append(
            "\n\nProvide responses in Markdown format. Be conversational, helpful, "
            "and personalize your responses based on the user's interests and context. "
            "If you need current information, use the web_search function."
        )
        
        return "".join(prompt_parts)

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str,
        user_context: UserContext,
        provider: Optional[str] = None
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        Generate AI response with function calling support.
        
        Main entry point for generating chat responses. Handles provider
        selection, function calling orchestration, and response formatting.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            system_prompt: System prompt for context
            user_context: User context for personalization
            provider: Optional provider override ("openai" or "gemini")
            
        Returns:
            Tuple of (response_text, updated_messages_with_function_calls)
            
        Requirements: 6.2, 6.3, 13.6
        """
        provider = provider or self.default_provider
        logger.info(
            f"Generating response: provider={provider}, message_count={len(messages)}\n"
            f"User context: chatInterest={user_context.chatInterest}, "
            f"has_summary={bool(user_context.userSummary)}, "
            f"history_length={len(user_context.chatHistory)}"
        )
        logger.debug(f"=== USER CONTEXT ===")
        logger.debug(f"Chat Interest: {user_context.chatInterest}")
        logger.debug(f"Topics: {user_context.topics}")
        logger.debug(f"Birthdate: {user_context.birthdate}")
        logger.debug(f"User Summary: {user_context.userSummary if user_context.userSummary else '(empty)'}")
        logger.debug(f"=== END USER CONTEXT ===")
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Call appropriate provider
            logger.debug(f"Calling {provider} provider...")
            if provider == "openai":
                response_text, function_call = await self._call_openai(
                    messages, system_prompt
                )
            elif provider == "gemini":
                response_text, function_call = await self._call_gemini(
                    messages, system_prompt
                )
            else:
                raise ValueError(f"Unsupported provider: {provider}")
            
            logger.debug(f"Provider call completed: has_function_call={bool(function_call)}, response_length={len(response_text)}")
            
            # Handle function calling if requested
            updated_messages = messages.copy()
            
            if function_call:
                logger.info(f"Function call requested: {function_call.get('name')} with args: {function_call.get('arguments')}")
                
                # Execute function
                function_result = await self._handle_function_call(
                    function_call.get("name"),
                    function_call.get("arguments", {})
                )
                logger.debug(f"Function result length: {len(function_result)}")
                
                # Add function call and result to messages
                updated_messages.append({
                    "role": "assistant",
                    "content": None,
                    "function_call": function_call
                })
                updated_messages.append({
                    "role": "function",
                    "name": function_call.get("name"),
                    "content": function_result
                })
                
                logger.debug(f"Generating final response with function results...")
                # Generate final response with function results
                if provider == "openai":
                    response_text, _ = await self._call_openai(
                        updated_messages, system_prompt
                    )
                else:
                    response_text, _ = await self._call_gemini(
                        updated_messages, system_prompt
                    )
            
            duration = asyncio.get_event_loop().time() - start_time
            logger.info(
                f"Response generated successfully: provider={provider}, duration={duration:.3f}s, "
                f"response_length={len(response_text)}, had_function_call={bool(function_call)}"
            )
            
            return response_text, updated_messages
            
        except Exception as e:
            duration = asyncio.get_event_loop().time() - start_time
            logger.error(
                f"Response generation FAILED: provider={provider}, duration={duration:.3f}s\n"
                f"Error type: {type(e).__name__}\n"
                f"Error message: {str(e)}\n"
                f"Message count: {len(messages)}\n"
                f"System prompt length: {len(system_prompt)}",
                exc_info=True
            )
            raise
    
    async def _call_openai(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Call OpenAI API with retry logic and function calling support.
        
        Args:
            messages: List of message dicts
            system_prompt: System prompt
            
        Returns:
            Tuple of (response_text, function_call_dict or None)
            
        Requirements: 6.2, 6.4, 13.6, 13.11, 11.3
        """
        logger.debug(f"Calling OpenAI API: message_count={len(messages)}")
        
        async def _make_openai_call():
            # Prepare messages with system prompt
            api_messages = [{"role": "system", "content": system_prompt}]
            api_messages.extend(messages)
            
            # Make API call with function calling support
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=api_messages,
                functions=[self.function_schema],
                function_call="auto",
                temperature=0.7
            )
            
            return response
        
        # Use retry logic
        response = await retry_with_backoff(
            _make_openai_call,
            max_retries=3,
            base_delay=1.0,
            max_delay=10.0
        )
        
        # Extract response
        choice = response.choices[0]
        message = choice.message
        
        # Log token usage
        if hasattr(response, 'usage'):
            logger.info(
                f"OpenAI API call: prompt_tokens={response.usage.prompt_tokens}, "
                f"completion_tokens={response.usage.completion_tokens}, "
                f"total_tokens={response.usage.total_tokens}"
            )
        
        # Check for function call
        function_call = None
        if hasattr(message, 'function_call') and message.function_call:
            import json
            function_call = {
                "name": message.function_call.name,
                "arguments": json.loads(message.function_call.arguments)
            }
            return "", function_call
        
        return message.content or "", None

    async def _call_gemini(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Call Google Gemini API with retry logic and function calling support.
        
        Args:
            messages: List of message dicts
            system_prompt: System prompt
            
        Returns:
            Tuple of (response_text, function_call_dict or None)
            
        Requirements: 6.2, 6.4, 13.6, 13.11, 11.3
        """
        logger.debug(f"Calling Gemini API: message_count={len(messages)}")
        logger.debug(f"System prompt length: {len(system_prompt)}")
        logger.debug(f"Messages structure: {[{k: v for k, v in msg.items() if k != 'content'} for msg in messages]}")
        
        async def _make_gemini_call():
            try:
                logger.debug("Building function declaration for Gemini")
                # Define function declaration using dict format
                # Define function declaration using Gemini's schema format
                # Gemini uses TYPE_STRING, TYPE_INTEGER instead of "string", "integer"
                function_declaration = {
                    "name": self.function_schema["name"],
                    "description": self.function_schema["description"],
                    "parameters": {
                        "type_": "OBJECT",
                        "properties": {
                            "query": {
                                "type_": "STRING",
                                "description": self.function_schema["parameters"]["properties"]["query"]["description"]
                            },
                            "count": {
                                "type_": "INTEGER",
                                "description": self.function_schema["parameters"]["properties"]["count"]["description"]
                            }
                        },
                        "required": ["query"]
                    }
                }
                logger.debug(f"Function declaration created: {function_declaration}")
                
                # Initialize model with function declarations
                logger.debug("Initializing Gemini model: models/gemini-2.5-flash")
                model = genai.GenerativeModel(
                    model_name="models/gemini-2.5-flash",
                    tools=[function_declaration]
                )
                logger.debug("Gemini model initialized successfully")
                
                # Convert messages to Gemini format, prepending system prompt as first user message
                gemini_messages = []
                
                # Add system prompt as first user message
                logger.debug("Adding system prompt to messages")
                gemini_messages.append({
                    "role": "user",
                    "parts": [system_prompt]
                })
                
                gemini_messages.append({
                    "role": "model",
                    "parts": ["Understood. I'll follow these instructions."]
                })
                # Add conversation messages
                logger.debug(f"Converting {len(messages)} messages to Gemini format")
                for idx, msg in enumerate(messages):
                    role = "user" if msg["role"] == "user" else "model"
                    content = msg.get("content", "")
                    logger.debug(f"Message {idx}: role={msg['role']}, has_content={bool(content)}, content_length={len(content) if content else 0}")
                    if content:  # Skip empty content
                        gemini_messages.append({
                            "role": role,
                            "parts": [content]
                        })
                
                logger.debug(f"Final Gemini messages count: {len(gemini_messages)}")
                logger.debug(f"Gemini messages structure: {[{'role': m['role'], 'parts_count': len(m.get('parts', []))} for m in gemini_messages]}")
                
                # Generate response
                logger.debug("Calling Gemini API generate_content_async...")
                response = await model.generate_content_async(
                    gemini_messages,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7
                    )
                )
                logger.debug("Gemini API call completed successfully")
                logger.debug(f"Response type: {type(response)}")
                logger.debug(f"Response attributes: {dir(response)}")
                
                return response
                
            except Exception as e:
                logger.error(
                    f"Error in _make_gemini_call: {type(e).__name__}: {str(e)}\n"
                    f"Messages count: {len(messages)}\n"
                    f"System prompt length: {len(system_prompt)}\n"
                    f"Gemini messages count: {len(gemini_messages) if 'gemini_messages' in locals() else 'not created yet'}"
                )
                raise
        
        # Use retry logic
        logger.debug("Starting retry_with_backoff for Gemini call")
        response = await retry_with_backoff(
            _make_gemini_call,
            max_retries=3,
            base_delay=1.0,
            max_delay=10.0,
            operation_name="_make_gemini_call"
        )
        
        logger.debug("Processing Gemini response...")
        
        # Log token usage if available
        try:
            if hasattr(response, 'usage_metadata'):
                logger.info(
                    f"Gemini API call: prompt_tokens={response.usage_metadata.prompt_token_count}, "
                    f"completion_tokens={response.usage_metadata.candidates_token_count}, "
                    f"total_tokens={response.usage_metadata.total_token_count}"
                )
            else:
                logger.debug("Response has no usage_metadata attribute")
        except Exception as e:
            logger.warning(f"Failed to log token usage: {e}")
        
        # Check for function call
        try:
            logger.debug(f"Checking for function calls in response")
            logger.debug(f"Response has candidates: {hasattr(response, 'candidates')}")
            if hasattr(response, 'candidates'):
                logger.debug(f"Candidates count: {len(response.candidates) if response.candidates else 0}")
                
            if response.candidates and response.candidates[0].content.parts:
                logger.debug(f"Parts count: {len(response.candidates[0].content.parts)}")
                for idx, part in enumerate(response.candidates[0].content.parts):
                    logger.debug(f"Part {idx} type: {type(part)}, attributes: {dir(part)}")
                    if hasattr(part, 'function_call') and part.function_call:
                        logger.info(f"Function call detected: {part.function_call.name}")
                        function_call = {
                            "name": part.function_call.name,
                            "arguments": dict(part.function_call.args)
                        }
                        logger.debug(f"Function call arguments: {function_call['arguments']}")
                        return "", function_call
        except Exception as e:
            logger.error(f"Error checking for function calls: {type(e).__name__}: {str(e)}", exc_info=True)
        
        # Extract text response
        try:
            logger.debug("Extracting text response from Gemini")
            if response.candidates and response.candidates[0].content.parts:
                text_parts = [
                    part.text for part in response.candidates[0].content.parts
                    if hasattr(part, 'text')
                ]
                logger.debug(f"Extracted {len(text_parts)} text parts, total length: {sum(len(p) for p in text_parts)}")
                result = "".join(text_parts)
                logger.debug(f"Final response length: {len(result)}")
                return result, None
            else:
                logger.warning("No candidates or parts found in response")
        except Exception as e:
            logger.error(f"Error extracting text response: {type(e).__name__}: {str(e)}", exc_info=True)
        
        logger.warning("Returning empty response from Gemini")
        return "", None
    
    async def _handle_function_call(
        self,
        function_name: str,
        arguments: Dict[str, Any]
    ) -> str:
        """
        Execute function calls (e.g., web search).
        
        Handles function call execution and formats results for LLM consumption.
        Currently supports web_search function.
        
        Args:
            function_name: Name of the function to call
            arguments: Function arguments dict
            
        Returns:
            Formatted function result as string
            
        Requirements: 6.3, 13.7
        """
        logger.info(f"Executing function: {function_name} with args: {arguments}")
        
        try:
            if function_name == "web_search":
                if not self.search_service:
                    logger.warning("Search service not available")
                    return "Search service is currently unavailable."
                
                query = arguments.get("query", "")
                count = arguments.get("count", 5)
                
                # Perform search
                results = await self.search_service.search(query, count)
                
                if not results:
                    return f"No search results found for query: {query}"
                
                # Format results for LLM
                formatted_results = [
                    f"**{i+1}. {result['title']}**\n"
                    f"URL: {result['url']}\n"
                    f"Description: {result['description']}\n"
                    for i, result in enumerate(results)
                ]
                
                return "Search results:\n\n" + "\n".join(formatted_results)
            
            else:
                logger.warning(f"Unknown function: {function_name}")
                return f"Unknown function: {function_name}"
                
        except Exception as e:
            logger.error(f"Function call failed: {function_name}, error: {e}")
            return f"Function call failed: {str(e)}"

    async def summarize_messages(
        self,
        messages: List[Dict[str, str]],
        previous_summary: Optional[str] = None,
        provider: Optional[str] = None
    ) -> str:
        """
        Summarize old messages for context compression.
        
        Creates a concise summary of conversation history to maintain context
        while staying within token limits. If a previous summary exists, it
        compresses both the previous summary and new messages into an updated
        summary.
        
        Args:
            messages: List of message dicts to summarize
            previous_summary: Optional previous conversation summary to include
            provider: Optional provider override
            
        Returns:
            Summary text
            
        Requirements: 5.2, 5.3, 13.9
        """
        provider = provider or self.default_provider
        logger.info(
            f"Summarizing messages: count={len(messages)}, "
            f"has_previous_summary={previous_summary is not None}, provider={provider}"
        )
        
        # Build summarization prompt
        system_prompt = (
            "You are a helpful assistant that creates concise summaries of conversations. "
            "Summarize the conversation, capturing key topics, user preferences, "
            "and important context. Keep the summary brief but informative."
        )
        
        # Format messages for summarization
        conversation_text = "\n".join([
            f"{msg['role'].upper()}: {msg.get('content', '')}"
            for msg in messages
            if msg.get('content')
        ])
        
        # Build prompt based on whether we have previous summary
        if previous_summary:
            prompt_content = (
                f"Previous conversation summary:\n{previous_summary}\n\n"
                f"New messages:\n{conversation_text}\n\n"
                f"Please create an updated summary that compresses both the previous "
                f"summary and the new messages into a single concise summary."
            )
        else:
            prompt_content = f"Please summarize this conversation:\n\n{conversation_text}"
        
        summarization_messages = [
            {
                "role": "user",
                "content": prompt_content
            }
        ]
        
        try:
            # Generate summary
            if provider == "openai":
                summary, _ = await self._call_openai(
                    summarization_messages,
                    system_prompt
                )
            else:
                summary, _ = await self._call_gemini(
                    summarization_messages,
                    system_prompt
                )
            
            logger.info(f"Summary generated: length={len(summary)}")
            return summary
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            # Return a basic summary on failure
            if previous_summary:
                return f"{previous_summary}\n\nAdditional {len(messages)} messages discussed."
            return f"Previous conversation covered {len(messages)} messages."
