"""
Unit tests for database service.
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from appwrite.exception import AppwriteException

from app.db_service import DatabaseService
from app.models import UserContext, Message


@pytest.fixture
def db_service():
    """Create a DatabaseService instance with test credentials."""
    return DatabaseService(
        endpoint="https://test.appwrite.io/v1",
        project_id="test_project",
        api_key="test_api_key",
        database_id="test_database",
        collection_id="test_collection"
    )


@pytest.fixture
def sample_user_context():
    """Create a sample UserContext for testing."""
    return UserContext(
        chatHistory=[
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!")
        ],
        chatInterest="Python programming",
        userSummary="User is interested in learning Python",
        birthdate="1990-01-01",
        topics=["Python", "AI"]
    )


@pytest.fixture
def mock_appwrite_document():
    """Create a mock Appwrite document response."""
    return {
        "$id": "user123",
        "chatHistory": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ],
        "chatInterest": "Python programming",
        "userSummary": "User is interested in learning Python",
        "birthdate": "1990-01-01",
        "topics": ["Python", "AI"],
        "$createdAt": "2024-01-01T00:00:00.000+00:00",
        "$updatedAt": "2024-01-01T00:00:00.000+00:00"
    }


class TestDatabaseService:
    """Tests for DatabaseService class."""
    
    def test_initialization(self, db_service):
        """Test database service initialization."""
        assert db_service.database_id == "test_database"
        assert db_service.collection_id == "test_collection"
        assert db_service.databases is not None
        assert db_service.client is not None

    
    @pytest.mark.asyncio
    async def test_get_user_context_success(self, db_service, mock_appwrite_document):
        """Test successful user context retrieval."""
        user_id = "user123"
        
        # Mock the get_document method
        with patch.object(db_service.databases, 'get_document', return_value=mock_appwrite_document):
            context = await db_service.get_user_context(user_id)
        
        assert context is not None
        assert isinstance(context, UserContext)
        assert len(context.chatHistory) == 2
        assert context.chatHistory[0].role == "user"
        assert context.chatHistory[0].content == "Hello"
        assert context.chatInterest == "Python programming"
        assert context.userSummary == "User is interested in learning Python"
        assert context.birthdate == "1990-01-01"
        assert context.topics == ["Python", "AI"]
    
    @pytest.mark.asyncio
    async def test_get_user_context_not_found(self, db_service):
        """Test user context retrieval when user doesn't exist."""
        user_id = "nonexistent_user"
        
        # Mock 404 exception
        mock_exception = AppwriteException("Document not found", 404, "not_found")
        
        with patch.object(db_service.databases, 'get_document', side_effect=mock_exception):
            context = await db_service.get_user_context(user_id)
        
        assert context is None
    
    @pytest.mark.asyncio
    async def test_get_user_context_with_empty_history(self, db_service):
        """Test user context retrieval with empty chat history."""
        user_id = "user_empty"
        
        mock_doc = {
            "$id": user_id,
            "chatHistory": [],
            "chatInterest": "Testing",
            "userSummary": "",
            "birthdate": None,
            "topics": []
        }
        
        with patch.object(db_service.databases, 'get_document', return_value=mock_doc):
            context = await db_service.get_user_context(user_id)
        
        assert context is not None
        assert len(context.chatHistory) == 0
        assert context.chatInterest == "Testing"
        assert context.userSummary == ""
        assert context.birthdate is None
        assert context.topics == []
    
    @pytest.mark.asyncio
    async def test_get_user_context_server_error(self, db_service):
        """Test user context retrieval with server error."""
        user_id = "user_error"
        
        # Mock 500 server error
        mock_exception = AppwriteException("Internal server error", 500, "server_error")
        
        with patch.object(db_service.databases, 'get_document', side_effect=mock_exception):
            with pytest.raises(AppwriteException) as exc_info:
                await db_service.get_user_context(user_id)
            
            assert exc_info.value.code == 500

    
    @pytest.mark.asyncio
    async def test_update_chat_history_success(self, db_service):
        """Test successful chat history update."""
        user_id = "user123"
        chat_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "How are you?"}
        ]
        user_summary = "Previous conversation summary"
        
        mock_response = {"$id": user_id, "chatHistory": chat_history, "userSummary": user_summary}
        
        with patch.object(db_service.databases, 'update_document', return_value=mock_response):
            # Should not raise any exception
            await db_service.update_chat_history(user_id, chat_history, user_summary)
    
    @pytest.mark.asyncio
    async def test_update_chat_history_without_summary(self, db_service):
        """Test chat history update without summary."""
        user_id = "user456"
        chat_history = [
            {"role": "user", "content": "Test message"}
        ]
        
        mock_response = {"$id": user_id, "chatHistory": chat_history, "userSummary": ""}
        
        with patch.object(db_service.databases, 'update_document', return_value=mock_response):
            await db_service.update_chat_history(user_id, chat_history)
    
    @pytest.mark.asyncio
    async def test_update_chat_history_error(self, db_service):
        """Test chat history update with error."""
        user_id = "user_error"
        chat_history = [{"role": "user", "content": "Test"}]
        
        # Mock 500 server error
        mock_exception = AppwriteException("Update failed", 500, "server_error")
        
        with patch.object(db_service.databases, 'update_document', side_effect=mock_exception):
            with pytest.raises(AppwriteException) as exc_info:
                await db_service.update_chat_history(user_id, chat_history)
            
            assert exc_info.value.code == 500
    
    @pytest.mark.asyncio
    async def test_create_user_context_success(self, db_service, sample_user_context):
        """Test successful user context creation."""
        user_id = "new_user"
        
        mock_response = {
            "$id": user_id,
            "chatHistory": [
                {"role": msg.role, "content": msg.content}
                for msg in sample_user_context.chatHistory
            ],
            "chatInterest": sample_user_context.chatInterest,
            "userSummary": sample_user_context.userSummary,
            "birthdate": sample_user_context.birthdate,
            "topics": sample_user_context.topics
        }
        
        with patch.object(db_service.databases, 'create_document', return_value=mock_response):
            # Should not raise any exception
            await db_service.create_user_context(user_id, sample_user_context)
    
    @pytest.mark.asyncio
    async def test_create_user_context_minimal(self, db_service):
        """Test user context creation with minimal data."""
        user_id = "minimal_user"
        context = UserContext(
            chatHistory=[],
            chatInterest=None,
            userSummary="",
            birthdate=None,
            topics=[]
        )
        
        mock_response = {
            "$id": user_id,
            "chatHistory": [],
            "chatInterest": None,
            "userSummary": "",
            "birthdate": None,
            "topics": []
        }
        
        with patch.object(db_service.databases, 'create_document', return_value=mock_response):
            await db_service.create_user_context(user_id, context)
    
    @pytest.mark.asyncio
    async def test_create_user_context_error(self, db_service, sample_user_context):
        """Test user context creation with error."""
        user_id = "error_user"
        
        # Mock 500 server error
        mock_exception = AppwriteException("Creation failed", 500, "server_error")
        
        with patch.object(db_service.databases, 'create_document', side_effect=mock_exception):
            with pytest.raises(AppwriteException) as exc_info:
                await db_service.create_user_context(user_id, sample_user_context)
            
            assert exc_info.value.code == 500

    
    @pytest.mark.asyncio
    async def test_retry_logic_on_transient_failure(self, db_service):
        """Test retry logic with transient failures."""
        user_id = "user_retry"
        
        # Create a mock that fails twice then succeeds
        call_count = 0
        def mock_get_document(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise AppwriteException("Temporary error", 503, "service_unavailable")
            return {
                "$id": user_id,
                "chatHistory": [],
                "chatInterest": "Test",
                "userSummary": "",
                "birthdate": None,
                "topics": []
            }
        
        with patch.object(db_service.databases, 'get_document', side_effect=mock_get_document):
            # Should succeed after retries
            context = await db_service.get_user_context(user_id)
            
            assert context is not None
            assert call_count == 3  # Failed twice, succeeded on third attempt
    
    @pytest.mark.asyncio
    async def test_retry_exhausted(self, db_service):
        """Test behavior when all retries are exhausted."""
        user_id = "user_retry_fail"
        
        # Mock that always fails
        mock_exception = AppwriteException("Persistent error", 503, "service_unavailable")
        
        with patch.object(db_service.databases, 'get_document', side_effect=mock_exception):
            with pytest.raises(AppwriteException) as exc_info:
                await db_service.get_user_context(user_id)
            
            assert exc_info.value.code == 503
    
    @pytest.mark.asyncio
    async def test_update_retry_on_failure(self, db_service):
        """Test retry logic for update operations."""
        user_id = "user_update_retry"
        chat_history = [{"role": "user", "content": "Test"}]
        
        # Create a mock that fails once then succeeds
        call_count = 0
        def mock_update_document(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise AppwriteException("Temporary error", 503, "service_unavailable")
            return {"$id": user_id, "chatHistory": chat_history}
        
        with patch.object(db_service.databases, 'update_document', side_effect=mock_update_document):
            # Should succeed after retry
            await db_service.update_chat_history(user_id, chat_history)
            
            assert call_count == 2  # Failed once, succeeded on second attempt
    
    @pytest.mark.asyncio
    async def test_create_retry_on_failure(self, db_service):
        """Test retry logic for create operations."""
        user_id = "user_create_retry"
        context = UserContext(chatHistory=[], chatInterest="Test")
        
        # Create a mock that fails once then succeeds
        call_count = 0
        def mock_create_document(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise AppwriteException("Temporary error", 503, "service_unavailable")
            return {"$id": user_id, "chatHistory": [], "chatInterest": "Test"}
        
        with patch.object(db_service.databases, 'create_document', side_effect=mock_create_document):
            # Should succeed after retry
            await db_service.create_user_context(user_id, context)
            
            assert call_count == 2  # Failed once, succeeded on second attempt
    
    @pytest.mark.asyncio
    async def test_get_user_context_with_missing_fields(self, db_service):
        """Test user context retrieval with missing optional fields."""
        user_id = "user_partial"
        
        # Document with only required fields
        mock_doc = {
            "$id": user_id,
            "chatHistory": [{"role": "user", "content": "Hi"}]
            # Missing: chatInterest, userSummary, birthdate, topics
        }
        
        with patch.object(db_service.databases, 'get_document', return_value=mock_doc):
            context = await db_service.get_user_context(user_id)
        
        assert context is not None
        assert len(context.chatHistory) == 1
        assert context.chatInterest is None
        assert context.userSummary == ""
        assert context.birthdate is None
        assert context.topics == []
