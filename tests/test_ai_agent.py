"""Unit tests for AI agent."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from app.ai_agent import AIAgent
from app.models import UserContext, Message


@pytest.fixture
def user_context_first_time():
    """Create a user context for first-time user."""
    return UserContext(
        chatHistory=[],
        chatInterest="Python programming",
        userSummary="",
        birthdate="1990-01-15",
        topics=["programming", "AI", "web development"]
    )


@pytest.fixture
def user_context_returning():
    """Create a user context for returning user."""
    return UserContext(
        chatHistory=[
            Message(role="user", content="Tell me about Python"),
            Message(role="assistant", content="Python is a versatile programming language...")
        ],
        chatInterest="Python programming",
        userSummary="User is learning Python and interested in web development.",
        birthdate="1990-01-15",
        topics=["programming", "AI", "web development"]
    )


@pytest.fixture
def mock_search_service():
    """Create a mock search service."""
    service = MagicMock()
    service.search = AsyncMock(return_value=[
        {
            "title": "Python Tutorial",
            "url": "https://example.com/python",
            "description": "Learn Python programming"
        },
        {
            "title": "Python Documentation",
            "url": "https://docs.python.org",
            "description": "Official Python docs"
        }
    ])
    return service


@pytest.fixture
def ai_agent(mock_search_service):
    """Create an AIAgent instance for testing."""
    return AIAgent(
        openai_key="test_openai_key",
        gemini_key="test_gemini_key",
        default_provider="openai",
        search_service=mock_search_service
    )


class TestAIAgent:
    """Tests for AIAgent class."""
    
    def test_initialization(self):
        """Test AI agent initialization."""
        agent = AIAgent(
            openai_key="test_openai",
            gemini_key="test_gemini",
            default_provider="gemini"
        )
        
        assert agent.default_provider == "gemini"
        assert agent.openai_client is not None
        assert agent.search_service is None
        assert agent.function_schema["name"] == "web_search"
    
    def test_initialization_with_search_service(self, mock_search_service):
        """Test AI agent initialization with search service."""
        agent = AIAgent(
            openai_key="test_openai",
            gemini_key="test_gemini",
            search_service=mock_search_service
        )
        
        assert agent.search_service is mock_search_service
    
    def test_build_system_prompt_first_time_user(self, ai_agent, user_context_first_time):
        """Test prompt building for first-time user."""
        prompt = ai_agent._build_system_prompt(user_context_first_time, is_first_message=True)
        
        assert "helpful and personalized AI assistant" in prompt
        assert "Python programming" in prompt
        assert "programming, AI, web development" in prompt
        assert "1990-01-15" in prompt
        assert "Previous conversation summary" not in prompt  # First-time user
    
    def test_build_system_prompt_returning_user(self, ai_agent, user_context_returning):
        """Test prompt building for returning user with summary."""
        prompt = ai_agent._build_system_prompt(user_context_returning, is_first_message=False)
        
        assert "helpful and personalized AI assistant" in prompt
        assert "Python programming" in prompt
        assert "Previous conversation summary" in prompt
        assert "User is learning Python" in prompt
        assert "Markdown format" in prompt
    
    def test_build_system_prompt_minimal_context(self, ai_agent):
        """Test prompt building with minimal user context."""
        minimal_context = UserContext(
            chatHistory=[],
            chatInterest=None,
            userSummary="",
            birthdate=None,
            topics=[]
        )
        
        prompt = ai_agent._build_system_prompt(minimal_context, is_first_message=True)
        
        assert "helpful and personalized AI assistant" in prompt
        assert "Markdown format" in prompt

    @pytest.mark.asyncio
    async def test_call_openai_success(self, ai_agent):
        """Test successful OpenAI API call."""
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = "This is a test response from OpenAI."
        mock_response.choices[0].message.function_call = None
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 70
        
        ai_agent.openai_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        messages = [{"role": "user", "content": "Hello"}]
        system_prompt = "You are a helpful assistant."
        
        response_text, function_call = await ai_agent._call_openai(messages, system_prompt)
        
        assert response_text == "This is a test response from OpenAI."
        assert function_call is None
        ai_agent.openai_client.chat.completions.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_call_openai_with_function_call(self, ai_agent):
        """Test OpenAI API call with function calling."""
        # Mock OpenAI response with function call
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = None
        mock_response.choices[0].message.function_call = MagicMock()
        mock_response.choices[0].message.function_call.name = "web_search"
        mock_response.choices[0].message.function_call.arguments = '{"query": "Python tutorials", "count": 5}'
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 60
        mock_response.usage.completion_tokens = 15
        mock_response.usage.total_tokens = 75
        
        ai_agent.openai_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        messages = [{"role": "user", "content": "Search for Python tutorials"}]
        system_prompt = "You are a helpful assistant."
        
        response_text, function_call = await ai_agent._call_openai(messages, system_prompt)
        
        assert response_text == ""
        assert function_call is not None
        assert function_call["name"] == "web_search"
        assert function_call["arguments"]["query"] == "Python tutorials"
        assert function_call["arguments"]["count"] == 5
    
    @pytest.mark.asyncio
    async def test_call_gemini_success(self, ai_agent):
        """Test successful Gemini API call."""
        # Mock Gemini response with proper text attribute
        mock_part = MagicMock()
        mock_part.text = "This is a test response from Gemini."
        # Ensure hasattr(part, 'text') returns True
        type(mock_part).text = property(lambda self: "This is a test response from Gemini.")
        # No function_call attribute
        mock_part.function_call = None
        
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content = MagicMock()
        mock_response.candidates[0].content.parts = [mock_part]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 45
        mock_response.usage_metadata.candidates_token_count = 18
        mock_response.usage_metadata.total_token_count = 63
        
        with patch('google.generativeai.GenerativeModel') as mock_model_class:
            mock_model = MagicMock()
            mock_model.generate_content_async = AsyncMock(return_value=mock_response)
            mock_model_class.return_value = mock_model
            
            messages = [{"role": "user", "content": "Hello"}]
            system_prompt = "You are a helpful assistant."
            
            response_text, function_call = await ai_agent._call_gemini(messages, system_prompt)
            
            assert response_text == "This is a test response from Gemini."
            assert function_call is None
    
    @pytest.mark.asyncio
    async def test_call_gemini_with_function_call(self, ai_agent):
        """Test Gemini API call with function calling."""
        # Mock Gemini response with function call
        mock_function_call = MagicMock()
        mock_function_call.name = "web_search"
        mock_function_call.args = {"query": "AI news", "count": 3}
        
        mock_part = MagicMock()
        mock_part.function_call = mock_function_call
        
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content = MagicMock()
        mock_response.candidates[0].content.parts = [mock_part]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 55
        mock_response.usage_metadata.candidates_token_count = 12
        mock_response.usage_metadata.total_token_count = 67
        
        with patch('google.generativeai.GenerativeModel') as mock_model_class:
            mock_model = MagicMock()
            mock_model.generate_content_async = AsyncMock(return_value=mock_response)
            mock_model_class.return_value = mock_model
            
            messages = [{"role": "user", "content": "Search for AI news"}]
            system_prompt = "You are a helpful assistant."
            
            response_text, function_call = await ai_agent._call_gemini(messages, system_prompt)
            
            assert response_text == ""
            assert function_call is not None
            assert function_call["name"] == "web_search"
            assert function_call["arguments"]["query"] == "AI news"

    @pytest.mark.asyncio
    async def test_handle_function_call_web_search(self, ai_agent, mock_search_service):
        """Test handling web search function call."""
        result = await ai_agent._handle_function_call(
            "web_search",
            {"query": "Python tutorials", "count": 2}
        )
        
        assert "Search results:" in result
        assert "Python Tutorial" in result
        assert "https://example.com/python" in result
        assert "Python Documentation" in result
        mock_search_service.search.assert_called_once_with("Python tutorials", 2)
    
    @pytest.mark.asyncio
    async def test_handle_function_call_no_search_service(self, ai_agent):
        """Test handling function call when search service is unavailable."""
        ai_agent.search_service = None
        
        result = await ai_agent._handle_function_call(
            "web_search",
            {"query": "test query"}
        )
        
        assert "unavailable" in result.lower()
    
    @pytest.mark.asyncio
    async def test_handle_function_call_empty_results(self, ai_agent, mock_search_service):
        """Test handling function call with empty search results."""
        mock_search_service.search = AsyncMock(return_value=[])
        
        result = await ai_agent._handle_function_call(
            "web_search",
            {"query": "nonexistent query"}
        )
        
        assert "No search results found" in result
    
    @pytest.mark.asyncio
    async def test_handle_function_call_unknown_function(self, ai_agent):
        """Test handling unknown function call."""
        result = await ai_agent._handle_function_call(
            "unknown_function",
            {"arg": "value"}
        )
        
        assert "Unknown function" in result
    
    @pytest.mark.asyncio
    async def test_handle_function_call_search_error(self, ai_agent, mock_search_service):
        """Test handling function call when search fails."""
        mock_search_service.search = AsyncMock(side_effect=Exception("Search API error"))
        
        result = await ai_agent._handle_function_call(
            "web_search",
            {"query": "test"}
        )
        
        assert "Function call failed" in result
    
    @pytest.mark.asyncio
    async def test_generate_response_openai(self, ai_agent, user_context_first_time):
        """Test generating response with OpenAI."""
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = "Hello! How can I help you?"
        mock_response.choices[0].message.function_call = None
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 40
        mock_response.usage.completion_tokens = 10
        mock_response.usage.total_tokens = 50
        
        ai_agent.openai_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        messages = [{"role": "user", "content": "Hello"}]
        system_prompt = "You are helpful."
        
        response_text, updated_messages = await ai_agent.generate_response(
            messages, system_prompt, user_context_first_time, provider="openai"
        )
        
        assert response_text == "Hello! How can I help you?"
        assert updated_messages == messages  # No function calls
    
    @pytest.mark.asyncio
    async def test_generate_response_gemini(self, ai_agent, user_context_first_time):
        """Test generating response with Gemini."""
        # Mock Gemini response with proper text attribute
        mock_part = MagicMock()
        type(mock_part).text = property(lambda self: "Gemini response here.")
        mock_part.function_call = None
        
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content = MagicMock()
        mock_response.candidates[0].content.parts = [mock_part]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 35
        mock_response.usage_metadata.candidates_token_count = 8
        mock_response.usage_metadata.total_token_count = 43
        
        with patch('google.generativeai.GenerativeModel') as mock_model_class:
            mock_model = MagicMock()
            mock_model.generate_content_async = AsyncMock(return_value=mock_response)
            mock_model_class.return_value = mock_model
            
            messages = [{"role": "user", "content": "Hello"}]
            system_prompt = "You are helpful."
            
            response_text, updated_messages = await ai_agent.generate_response(
                messages, system_prompt, user_context_first_time, provider="gemini"
            )
            
            assert response_text == "Gemini response here."
            assert updated_messages == messages

    @pytest.mark.asyncio
    async def test_generate_response_with_function_calling(self, ai_agent, user_context_first_time, mock_search_service):
        """Test generating response with function calling flow."""
        # First call returns function call
        mock_response_1 = MagicMock()
        mock_response_1.choices = [MagicMock()]
        mock_response_1.choices[0].message = MagicMock()
        mock_response_1.choices[0].message.content = None
        mock_response_1.choices[0].message.function_call = MagicMock()
        mock_response_1.choices[0].message.function_call.name = "web_search"
        mock_response_1.choices[0].message.function_call.arguments = '{"query": "Python", "count": 2}'
        mock_response_1.usage = MagicMock()
        mock_response_1.usage.prompt_tokens = 50
        mock_response_1.usage.completion_tokens = 15
        mock_response_1.usage.total_tokens = 65
        
        # Second call returns final response
        mock_response_2 = MagicMock()
        mock_response_2.choices = [MagicMock()]
        mock_response_2.choices[0].message = MagicMock()
        mock_response_2.choices[0].message.content = "Based on search results, Python is great!"
        mock_response_2.choices[0].message.function_call = None
        mock_response_2.usage = MagicMock()
        mock_response_2.usage.prompt_tokens = 100
        mock_response_2.usage.completion_tokens = 25
        mock_response_2.usage.total_tokens = 125
        
        ai_agent.openai_client.chat.completions.create = AsyncMock(
            side_effect=[mock_response_1, mock_response_2]
        )
        
        messages = [{"role": "user", "content": "Tell me about Python"}]
        system_prompt = "You are helpful."
        
        response_text, updated_messages = await ai_agent.generate_response(
            messages, system_prompt, user_context_first_time, provider="openai"
        )
        
        assert response_text == "Based on search results, Python is great!"
        assert len(updated_messages) > len(messages)  # Function call added
        mock_search_service.search.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_response_default_provider(self, ai_agent, user_context_first_time):
        """Test generating response uses default provider."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = "Default provider response."
        mock_response.choices[0].message.function_call = None
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 30
        mock_response.usage.completion_tokens = 8
        mock_response.usage.total_tokens = 38
        
        ai_agent.openai_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        messages = [{"role": "user", "content": "Test"}]
        system_prompt = "You are helpful."
        
        # Don't specify provider - should use default (openai)
        response_text, _ = await ai_agent.generate_response(
            messages, system_prompt, user_context_first_time
        )
        
        assert response_text == "Default provider response."
        ai_agent.openai_client.chat.completions.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_response_invalid_provider(self, ai_agent, user_context_first_time):
        """Test generating response with invalid provider."""
        messages = [{"role": "user", "content": "Test"}]
        system_prompt = "You are helpful."
        
        with pytest.raises(ValueError, match="Unsupported provider"):
            await ai_agent.generate_response(
                messages, system_prompt, user_context_first_time, provider="invalid"
            )
    
    @pytest.mark.asyncio
    async def test_summarize_messages_openai(self, ai_agent):
        """Test message summarization with OpenAI."""
        messages = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
            {"role": "user", "content": "Tell me more."},
            {"role": "assistant", "content": "It's versatile and easy to learn."}
        ]
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = "User asked about Python. Assistant explained it's a versatile programming language."
        mock_response.choices[0].message.function_call = None
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 80
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 100
        
        ai_agent.openai_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        summary = await ai_agent.summarize_messages(messages, provider="openai")
        
        assert "Python" in summary
        assert "programming language" in summary
        ai_agent.openai_client.chat.completions.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_summarize_messages_gemini(self, ai_agent):
        """Test message summarization with Gemini."""
        messages = [
            {"role": "user", "content": "Explain AI"},
            {"role": "assistant", "content": "AI is artificial intelligence."}
        ]
        
        # Mock Gemini response with proper text attribute
        mock_part = MagicMock()
        type(mock_part).text = property(lambda self: "User learned about AI basics.")
        mock_part.function_call = None
        
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content = MagicMock()
        mock_response.candidates[0].content.parts = [mock_part]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 60
        mock_response.usage_metadata.candidates_token_count = 15
        mock_response.usage_metadata.total_token_count = 75
        
        with patch('google.generativeai.GenerativeModel') as mock_model_class:
            mock_model = MagicMock()
            mock_model.generate_content_async = AsyncMock(return_value=mock_response)
            mock_model_class.return_value = mock_model
            
            summary = await ai_agent.summarize_messages(messages, provider="gemini")
            
            assert "AI" in summary
            assert "basics" in summary

    @pytest.mark.asyncio
    async def test_summarize_messages_failure(self, ai_agent):
        """Test summarization handles failures gracefully."""
        messages = [
            {"role": "user", "content": "Test message"}
        ]
        
        ai_agent.openai_client.chat.completions.create = AsyncMock(
            side_effect=Exception("API error")
        )
        
        summary = await ai_agent.summarize_messages(messages, provider="openai")
        
        # Should return basic summary on failure
        assert "1 messages" in summary
    
    @pytest.mark.asyncio
    async def test_retry_logic_on_api_failure(self, ai_agent, user_context_first_time):
        """Test retry logic when API calls fail."""
        # First two calls fail, third succeeds
        call_count = 0
        
        async def mock_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary API error")
            
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message = MagicMock()
            mock_response.choices[0].message.content = "Success after retries"
            mock_response.choices[0].message.function_call = None
            mock_response.usage = MagicMock()
            mock_response.usage.prompt_tokens = 40
            mock_response.usage.completion_tokens = 10
            mock_response.usage.total_tokens = 50
            return mock_response
        
        ai_agent.openai_client.chat.completions.create = mock_create
        
        messages = [{"role": "user", "content": "Test"}]
        system_prompt = "You are helpful."
        
        response_text, _ = await ai_agent.generate_response(
            messages, system_prompt, user_context_first_time, provider="openai"
        )
        
        assert response_text == "Success after retries"
        assert call_count == 3  # Verify retries occurred
    
    @pytest.mark.asyncio
    async def test_retry_logic_exhausted(self, ai_agent, user_context_first_time):
        """Test behavior when all retries are exhausted."""
        ai_agent.openai_client.chat.completions.create = AsyncMock(
            side_effect=Exception("Persistent API error")
        )
        
        messages = [{"role": "user", "content": "Test"}]
        system_prompt = "You are helpful."
        
        with pytest.raises(Exception, match="Persistent API error"):
            await ai_agent.generate_response(
                messages, system_prompt, user_context_first_time, provider="openai"
            )
    
    @pytest.mark.asyncio
    async def test_empty_messages_list(self, ai_agent, user_context_first_time):
        """Test handling empty messages list."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = "Response to empty messages"
        mock_response.choices[0].message.function_call = None
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 20
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 25
        
        ai_agent.openai_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        messages = []
        system_prompt = "You are helpful."
        
        response_text, _ = await ai_agent.generate_response(
            messages, system_prompt, user_context_first_time, provider="openai"
        )
        
        assert response_text == "Response to empty messages"
    
    @pytest.mark.asyncio
    async def test_multiple_text_parts_gemini(self, ai_agent, user_context_first_time):
        """Test Gemini response with multiple text parts."""
        # Mock Gemini response with multiple text parts
        mock_part1 = MagicMock()
        type(mock_part1).text = property(lambda self: "First part. ")
        mock_part1.function_call = None
        
        mock_part2 = MagicMock()
        type(mock_part2).text = property(lambda self: "Second part.")
        mock_part2.function_call = None
        
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content = MagicMock()
        mock_response.candidates[0].content.parts = [mock_part1, mock_part2]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 30
        mock_response.usage_metadata.candidates_token_count = 10
        mock_response.usage_metadata.total_token_count = 40
        
        with patch('google.generativeai.GenerativeModel') as mock_model_class:
            mock_model = MagicMock()
            mock_model.generate_content_async = AsyncMock(return_value=mock_response)
            mock_model_class.return_value = mock_model
            
            messages = [{"role": "user", "content": "Test"}]
            system_prompt = "You are helpful."
            
            response_text, _ = await ai_agent.generate_response(
                messages, system_prompt, user_context_first_time, provider="gemini"
            )
            
            assert response_text == "First part. Second part."
    
    def test_function_schema_structure(self, ai_agent):
        """Test that function schema is properly structured."""
        schema = ai_agent.function_schema
        
        assert schema["name"] == "web_search"
        assert "description" in schema
        assert "parameters" in schema
        assert schema["parameters"]["type"] == "object"
        assert "query" in schema["parameters"]["properties"]
        assert "count" in schema["parameters"]["properties"]
        assert "query" in schema["parameters"]["required"]
