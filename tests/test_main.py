"""
Integration tests for main FastAPI chat endpoint.

Requirements: 14.2, 14.3, 14.6, 14.8
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
from app.models import UserContext, Message


@pytest.fixture
def mock_cache_manager():
    """Mock CacheManager for testing."""
    mock = AsyncMock()
    mock.get = AsyncMock(return_value=None)
    mock.set = AsyncMock()
    mock.check_and_summarize = AsyncMock(return_value=(False, None))
    mock.close = AsyncMock()
    return mock


@pytest.fixture
def mock_db_service():
    """Mock DatabaseService for testing."""
    mock = AsyncMock()
    mock.get_user_context = AsyncMock(return_value=None)
    mock.create_user_context = AsyncMock()
    mock.update_chat_history = AsyncMock()
    return mock


@pytest.fixture
def mock_search_service():
    """Mock SearchService for testing."""
    mock = AsyncMock()
    mock.search = AsyncMock(return_value=[
        {
            "title": "Test Result 1",
            "url": "https://example.com/1",
            "description": "Test description 1"
        },
        {
            "title": "Test Result 2",
            "url": "https://example.com/2",
            "description": "Test description 2"
        }
    ])
    mock.close = AsyncMock()
    return mock


@pytest.fixture
def mock_ai_agent():
    """Mock AIAgent for testing."""
    mock = AsyncMock()
    mock._build_system_prompt = MagicMock(return_value="Test system prompt")
    mock.generate_response = AsyncMock(
        return_value=("This is a test AI response", [])
    )
    mock.summarize_messages = AsyncMock(
        return_value="This is a test summary"
    )
    return mock


@pytest.fixture
def mock_settings():
    """Mock Settings for testing."""
    mock = MagicMock()
    mock.previous_message_context_length = 10
    mock.overlap_count = 5
    mock.cache_ttl_seconds = 600
    mock.default_llm_provider = "openai"
    mock.log_level = "INFO"
    mock.log_rotation = "100 MB"
    mock.log_retention = "30 days"
    return mock


@pytest.fixture
def test_client(mock_cache_manager, mock_db_service, mock_ai_agent, mock_search_service, mock_settings):
    """Create test client with mocked services."""
    with patch('main.cache_manager', mock_cache_manager), \
         patch('main.db_service', mock_db_service), \
         patch('main.ai_agent', mock_ai_agent), \
         patch('main.search_service', mock_search_service), \
         patch('main.get_settings', return_value=mock_settings):
        
        from main import app
        client = TestClient(app)
        yield client


class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    def test_health_check(self, test_client):
        """Test health check endpoint returns healthy status."""
        response = test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "chat-agent"
        assert data["version"] == "1.0.0"


class TestChatEndpoint:
    """Tests for chat endpoint."""
    
    def test_first_time_user_flow(self, test_client, mock_cache_manager, mock_db_service, mock_ai_agent):
        """Test first-time user interaction flow."""
        # Setup: cache miss, no user in DB
        mock_cache_manager.get.return_value = None
        mock_db_service.get_user_context.return_value = None
        mock_ai_agent.generate_response.return_value = (
            "Welcome! I'd love to help you learn about Python programming.",
            []
        )
        
        # Make request
        response = test_client.post("/chat", json={
            "userId": "new_user_123",
            "userMessage": "I want to learn Python",
            "chatInterest": True,
            "interestTopic": "Python programming"
        })
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "Python" in data["response"]
        
        # Verify cache was checked
        mock_cache_manager.get.assert_called_once_with("new_user_123")
        
        # Verify DB was checked
        mock_db_service.get_user_context.assert_called()
        
        # Verify AI agent was called
        mock_ai_agent.generate_response.assert_called_once()
        
        # Verify user was created in DB
        mock_db_service.create_user_context.assert_called_once()
        
        # Verify cache was updated
        mock_cache_manager.set.assert_called()
    
    def test_returning_user_flow(self, test_client, mock_cache_manager, mock_db_service, mock_ai_agent):
        """Test returning user interaction flow."""
        # Setup: user exists in cache
        existing_context = UserContext(
            chatHistory=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi! How can I help?")
            ],
            chatInterest="Python programming",
            userSummary="",
            birthdate="1990-01-01",
            topics=["Python", "AI"]
        )
        mock_cache_manager.get.return_value = existing_context
        # Mock that user exists in DB (for the check before update)
        mock_db_service.get_user_context.return_value = existing_context
        mock_ai_agent.generate_response.return_value = (
            "Sure! Python is great for beginners.",
            []
        )
        
        # Make request
        response = test_client.post("/chat", json={
            "userId": "existing_user_456",
            "userMessage": "Tell me more about Python",
            "chatInterest": False
        })
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "Python" in data["response"]
        
        # Verify cache hit
        mock_cache_manager.get.assert_called_once_with("existing_user_456")
        
        # Verify AI agent was called with history
        mock_ai_agent.generate_response.assert_called_once()
        
        # Verify chat history was updated in DB
        mock_db_service.update_chat_history.assert_called_once()
    
    def test_cache_miss_scenario(self, test_client, mock_cache_manager, mock_db_service, mock_ai_agent):
        """Test cache miss with user existing in database."""
        # Setup: cache miss, but user exists in DB
        mock_cache_manager.get.return_value = None
        
        db_context = UserContext(
            chatHistory=[
                Message(role="user", content="Previous message"),
                Message(role="assistant", content="Previous response")
            ],
            chatInterest="Machine Learning",
            userSummary="User is learning ML",
            birthdate="1995-05-15",
            topics=["ML", "Data Science"]
        )
        mock_db_service.get_user_context.return_value = db_context
        mock_ai_agent.generate_response.return_value = (
            "Let's continue learning about ML!",
            []
        )
        
        # Make request
        response = test_client.post("/chat", json={
            "userId": "cache_miss_user",
            "userMessage": "Continue our ML discussion",
            "chatInterest": False
        })
        
        # Verify response
        assert response.status_code == 200
        
        # Verify cache miss
        mock_cache_manager.get.assert_called_once()
        
        # Verify DB fetch
        mock_db_service.get_user_context.assert_called()
        
        # Verify context was cached after DB fetch
        assert mock_cache_manager.set.call_count >= 1
    
    def test_function_calling_flow(self, test_client, mock_cache_manager, mock_db_service, mock_ai_agent, mock_search_service):
        """Test function calling with web search."""
        # Setup: AI agent triggers function call
        mock_cache_manager.get.return_value = None
        mock_db_service.get_user_context.return_value = None
        
        # First call returns function call, second returns final response
        mock_ai_agent.generate_response.return_value = (
            "Based on recent news, Python 3.12 was released with new features.",
            []
        )
        
        # Make request that should trigger search
        response = test_client.post("/chat", json={
            "userId": "search_user",
            "userMessage": "What's new in Python?",
            "chatInterest": False
        })
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
    
    def test_validation_error_missing_interest_topic(self, test_client):
        """Test validation error when interestTopic is missing for first-time user."""
        response = test_client.post("/chat", json={
            "userId": "user123",
            "userMessage": "Hello",
            "chatInterest": True
            # Missing interestTopic
        })
        
        # Should return 422 validation error
        assert response.status_code == 422
    
    def test_validation_error_empty_user_id(self, test_client):
        """Test validation error for empty userId."""
        response = test_client.post("/chat", json={
            "userId": "",
            "userMessage": "Hello",
            "chatInterest": False
        })
        
        # Should return 422 validation error
        assert response.status_code == 422
    
    def test_validation_error_empty_message(self, test_client):
        """Test validation error for empty userMessage."""
        response = test_client.post("/chat", json={
            "userId": "user123",
            "userMessage": "",
            "chatInterest": False
        })
        
        # Should return 422 validation error
        assert response.status_code == 422
    
    def test_error_handling_db_failure(self, test_client, mock_cache_manager, mock_db_service):
        """Test error handling when database operation fails."""
        # Setup: cache miss, DB throws error
        mock_cache_manager.get.return_value = None
        mock_db_service.get_user_context.side_effect = Exception("Database connection failed")
        
        # Make request
        response = test_client.post("/chat", json={
            "userId": "error_user",
            "userMessage": "Hello",
            "chatInterest": False
        })
        
        # Should return 500 error
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
    
    def test_error_handling_ai_failure(self, test_client, mock_cache_manager, mock_db_service, mock_ai_agent):
        """Test error handling when AI generation fails."""
        # Setup: AI agent throws error
        mock_cache_manager.get.return_value = None
        mock_db_service.get_user_context.return_value = None
        mock_ai_agent.generate_response.side_effect = Exception("AI service unavailable")
        
        # Make request
        response = test_client.post("/chat", json={
            "userId": "ai_error_user",
            "userMessage": "Hello",
            "chatInterest": False
        })
        
        # Should return 500 error
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
    
    def test_summarization_trigger(self, test_client, mock_cache_manager, mock_db_service, mock_ai_agent):
        """Test that summarization is triggered when message count exceeds threshold."""
        # Setup: user with many messages
        messages = [
            Message(role="user" if i % 2 == 0 else "assistant", content=f"Message {i}")
            for i in range(20)
        ]
        existing_context = UserContext(
            chatHistory=messages,
            chatInterest="Testing",
            userSummary="",
            birthdate=None,
            topics=[]
        )
        
        mock_cache_manager.get.return_value = existing_context
        mock_cache_manager.check_and_summarize.return_value = (True, existing_context)
        mock_db_service.get_user_context.return_value = existing_context
        mock_ai_agent.generate_response.return_value = ("Response", [])
        mock_ai_agent.summarize_messages.return_value = "Summary of old messages"
        
        # Make request
        response = test_client.post("/chat", json={
            "userId": "summarize_user",
            "userMessage": "New message",
            "chatInterest": False
        })
        
        # Verify response
        assert response.status_code == 200
        
        # Verify summarization was triggered
        mock_ai_agent.summarize_messages.assert_called_once()


class TestSummarizationLogic:
    """Integration tests for summarization logic.
    
    Requirements: 14.2, 14.7
    """
    
    @pytest.fixture
    def summarization_test_client(self, mock_cache_manager, mock_db_service, mock_ai_agent, mock_search_service, mock_settings):
        """Create test client with mocked services for summarization tests."""
        with patch('main.cache_manager', mock_cache_manager), \
             patch('main.db_service', mock_db_service), \
             patch('main.ai_agent', mock_ai_agent), \
             patch('main.search_service', mock_search_service), \
             patch('main.get_settings', return_value=mock_settings), \
             patch('main.limiter.enabled', False):  # Disable rate limiting for tests
            
            from main import app
            client = TestClient(app)
            yield client
    
    def test_summarization_with_various_message_counts(self, summarization_test_client, mock_cache_manager, mock_db_service, mock_ai_agent, mock_settings):
        """Test summarization trigger with various message counts."""
        # Test case 1: Below threshold (10 + 5 = 15) - no summarization
        messages_below = [
            Message(role="user" if i % 2 == 0 else "assistant", content=f"Message {i}")
            for i in range(14)
        ]
        context_below = UserContext(
            chatHistory=messages_below,
            chatInterest="Testing",
            userSummary="",
            birthdate=None,
            topics=[]
        )
        
        mock_cache_manager.get.return_value = context_below
        mock_cache_manager.check_and_summarize.return_value = (False, context_below)
        mock_db_service.get_user_context.return_value = context_below
        mock_ai_agent.generate_response.return_value = ("Response", [])
        
        response = summarization_test_client.post("/chat", json={
            "userId": "user_below_threshold",
            "userMessage": "New message",
            "chatInterest": False
        })
        
        assert response.status_code == 200
        # Summarization should not be called
        mock_ai_agent.summarize_messages.assert_not_called()
        
        # Reset mocks
        mock_ai_agent.reset_mock()
        
        # Test case 2: At threshold (15 messages) - no summarization yet
        messages_at = [
            Message(role="user" if i % 2 == 0 else "assistant", content=f"Message {i}")
            for i in range(15)
        ]
        context_at = UserContext(
            chatHistory=messages_at,
            chatInterest="Testing",
            userSummary="",
            birthdate=None,
            topics=[]
        )
        
        mock_cache_manager.get.return_value = context_at
        mock_cache_manager.check_and_summarize.return_value = (False, context_at)
        mock_db_service.get_user_context.return_value = context_at
        
        response = summarization_test_client.post("/chat", json={
            "userId": "user_at_threshold",
            "userMessage": "New message",
            "chatInterest": False
        })
        
        assert response.status_code == 200
        mock_ai_agent.summarize_messages.assert_not_called()
        
        # Reset mocks
        mock_ai_agent.reset_mock()
        
        # Test case 3: Above threshold (16+ messages) - summarization triggered
        messages_above = [
            Message(role="user" if i % 2 == 0 else "assistant", content=f"Message {i}")
            for i in range(16)
        ]
        context_above = UserContext(
            chatHistory=messages_above,
            chatInterest="Testing",
            userSummary="",
            birthdate=None,
            topics=[]
        )
        
        mock_cache_manager.get.return_value = context_above
        mock_cache_manager.check_and_summarize.return_value = (True, context_above)
        mock_db_service.get_user_context.return_value = context_above
        mock_ai_agent.summarize_messages.return_value = "Summary of messages"
        
        response = summarization_test_client.post("/chat", json={
            "userId": "user_above_threshold",
            "userMessage": "New message",
            "chatInterest": False
        })
        
        assert response.status_code == 200
        # Summarization should be called
        mock_ai_agent.summarize_messages.assert_called_once()
    
    def test_summary_generation(self, summarization_test_client, mock_cache_manager, mock_db_service, mock_ai_agent):
        """Test that summary is properly generated and stored."""
        # Setup: user with messages exceeding threshold
        messages = [
            Message(role="user" if i % 2 == 0 else "assistant", content=f"Message {i}")
            for i in range(20)
        ]
        existing_context = UserContext(
            chatHistory=messages,
            chatInterest="Testing",
            userSummary="",
            birthdate=None,
            topics=[]
        )
        
        mock_cache_manager.get.return_value = existing_context
        mock_cache_manager.check_and_summarize.return_value = (True, existing_context)
        mock_db_service.get_user_context.return_value = existing_context
        mock_ai_agent.generate_response.return_value = ("New response", [])
        
        expected_summary = "This is a generated summary of the conversation"
        mock_ai_agent.summarize_messages.return_value = expected_summary
        
        # Make request
        response = summarization_test_client.post("/chat", json={
            "userId": "summary_gen_user",
            "userMessage": "Continue conversation",
            "chatInterest": False
        })
        
        assert response.status_code == 200
        
        # Verify summarize_messages was called with correct messages
        mock_ai_agent.summarize_messages.assert_called_once()
        call_args = mock_ai_agent.summarize_messages.call_args[0][0]
        assert isinstance(call_args, list)
        assert len(call_args) > 0
        
        # Verify summary was stored in database
        mock_db_service.update_chat_history.assert_called_once()
        update_call_kwargs = mock_db_service.update_chat_history.call_args[1]
        assert update_call_kwargs['user_summary'] == expected_summary
    
    def test_history_trimming(self, summarization_test_client, mock_cache_manager, mock_db_service, mock_ai_agent, mock_settings):
        """Test that chat history is properly trimmed after summarization."""
        # Setup: user with 25 messages (exceeds threshold of 15)
        messages = [
            Message(role="user" if i % 2 == 0 else "assistant", content=f"Message {i}")
            for i in range(25)
        ]
        existing_context = UserContext(
            chatHistory=messages,
            chatInterest="Testing",
            userSummary="",
            birthdate=None,
            topics=[]
        )
        
        mock_cache_manager.get.return_value = existing_context
        mock_cache_manager.check_and_summarize.return_value = (True, existing_context)
        mock_db_service.get_user_context.return_value = existing_context
        mock_ai_agent.generate_response.return_value = ("Response", [])
        mock_ai_agent.summarize_messages.return_value = "Summary"
        
        # Make request
        response = summarization_test_client.post("/chat", json={
            "userId": "trim_user",
            "userMessage": "New message",
            "chatInterest": False
        })
        
        assert response.status_code == 200
        
        # Verify that update_chat_history was called
        mock_db_service.update_chat_history.assert_called_once()
        
        # Get the chat_history argument from the call
        call_kwargs = mock_db_service.update_chat_history.call_args[1]
        trimmed_history = call_kwargs['chat_history']
        
        # After adding 2 new messages (user + assistant) and trimming to 10,
        # we should have exactly 10 messages
        assert len(trimmed_history) == mock_settings.previous_message_context_length
    
    def test_summary_persistence(self, summarization_test_client, mock_cache_manager, mock_db_service, mock_ai_agent):
        """Test that summary is persisted in both cache and database."""
        # Setup: user with existing summary
        messages = [
            Message(role="user" if i % 2 == 0 else "assistant", content=f"Message {i}")
            for i in range(20)
        ]
        existing_context = UserContext(
            chatHistory=messages,
            chatInterest="Testing",
            userSummary="Previous summary",
            birthdate=None,
            topics=[]
        )
        
        mock_cache_manager.get.return_value = existing_context
        mock_cache_manager.check_and_summarize.return_value = (True, existing_context)
        mock_db_service.get_user_context.return_value = existing_context
        mock_ai_agent.generate_response.return_value = ("Response", [])
        
        new_summary = "New summary section"
        mock_ai_agent.summarize_messages.return_value = new_summary
        
        # Make request
        response = summarization_test_client.post("/chat", json={
            "userId": "persist_user",
            "userMessage": "Message",
            "chatInterest": False
        })
        
        assert response.status_code == 200
        
        # Verify cache was updated with combined summary
        mock_cache_manager.set.assert_called()
        cache_call_args = mock_cache_manager.set.call_args[0]
        cached_context = cache_call_args[1]
        assert "Previous summary" in cached_context.userSummary
        assert new_summary in cached_context.userSummary
        
        # Verify database was updated with combined summary
        mock_db_service.update_chat_history.assert_called_once()
        db_call_kwargs = mock_db_service.update_chat_history.call_args[1]
        assert "Previous summary" in db_call_kwargs['user_summary']
        assert new_summary in db_call_kwargs['user_summary']
    
    def test_summarization_with_empty_initial_summary(self, summarization_test_client, mock_cache_manager, mock_db_service, mock_ai_agent):
        """Test summarization when user has no existing summary."""
        messages = [
            Message(role="user" if i % 2 == 0 else "assistant", content=f"Message {i}")
            for i in range(20)
        ]
        existing_context = UserContext(
            chatHistory=messages,
            chatInterest="Testing",
            userSummary="",  # Empty summary
            birthdate=None,
            topics=[]
        )
        
        mock_cache_manager.get.return_value = existing_context
        mock_cache_manager.check_and_summarize.return_value = (True, existing_context)
        mock_db_service.get_user_context.return_value = existing_context
        mock_ai_agent.generate_response.return_value = ("Response", [])
        
        new_summary = "First summary"
        mock_ai_agent.summarize_messages.return_value = new_summary
        
        # Make request
        response = summarization_test_client.post("/chat", json={
            "userId": "empty_summary_user",
            "userMessage": "Message",
            "chatInterest": False
        })
        
        assert response.status_code == 200
        
        # Verify summary is set (not appended)
        mock_db_service.update_chat_history.assert_called_once()
        db_call_kwargs = mock_db_service.update_chat_history.call_args[1]
        assert db_call_kwargs['user_summary'] == new_summary
