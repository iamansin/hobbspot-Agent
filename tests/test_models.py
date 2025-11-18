"""
Unit tests for Pydantic models.
"""
import pytest
from pydantic import ValidationError
from app.models import ChatRequest, ChatResponse, Message, UserContext


class TestChatRequest:
    """Tests for ChatRequest model."""
    
    def test_valid_first_time_user_request(self):
        """Test valid request for first-time user with chatInterest=true."""
        request = ChatRequest(
            userId="user123",
            userMessage="Hello",
            chatInterest=True,
            interestTopic="Python programming"
        )
        assert request.userId == "user123"
        assert request.userMessage == "Hello"
        assert request.chatInterest is True
        assert request.interestTopic == "Python programming"
    
    def test_valid_returning_user_request(self):
        """Test valid request for returning user with chatInterest=false."""
        request = ChatRequest(
            userId="user456",
            userMessage="How are you?",
            chatInterest=False
        )
        assert request.userId == "user456"
        assert request.userMessage == "How are you?"
        assert request.chatInterest is False
        assert request.interestTopic is None
    
    def test_missing_interest_topic_when_chat_interest_true(self):
        """Test validation error when interestTopic is missing but chatInterest is true."""
        with pytest.raises(ValidationError) as exc_info:
            ChatRequest(
                userId="user789",
                userMessage="Hi",
                chatInterest=True
            )
        errors = exc_info.value.errors()
        assert any('interestTopic' in str(error) for error in errors)
    
    def test_empty_user_id(self):
        """Test validation error for empty userId."""
        with pytest.raises(ValidationError) as exc_info:
            ChatRequest(
                userId="",
                userMessage="Hello",
                chatInterest=False
            )
        errors = exc_info.value.errors()
        assert any('userId' in str(error) for error in errors)
    
    def test_empty_user_message(self):
        """Test validation error for empty userMessage."""
        with pytest.raises(ValidationError) as exc_info:
            ChatRequest(
                userId="user123",
                userMessage="",
                chatInterest=False
            )
        errors = exc_info.value.errors()
        assert any('userMessage' in str(error) for error in errors)
    
    def test_missing_required_fields(self):
        """Test validation error when required fields are missing."""
        with pytest.raises(ValidationError):
            ChatRequest(userId="user123")


class TestChatResponse:
    """Tests for ChatResponse model."""
    
    def test_valid_response(self):
        """Test valid ChatResponse creation."""
        response = ChatResponse(response="This is a test response")
        assert response.response == "This is a test response"
    
    def test_empty_response(self):
        """Test that empty response string is allowed."""
        response = ChatResponse(response="")
        assert response.response == ""


class TestMessage:
    """Tests for Message model."""
    
    def test_valid_user_message(self):
        """Test valid message with role='user'."""
        message = Message(role="user", content="Hello there")
        assert message.role == "user"
        assert message.content == "Hello there"
    
    def test_valid_assistant_message(self):
        """Test valid message with role='assistant'."""
        message = Message(role="assistant", content="Hi! How can I help?")
        assert message.role == "assistant"
        assert message.content == "Hi! How can I help?"
    
    def test_invalid_role(self):
        """Test validation error for invalid role."""
        with pytest.raises(ValidationError) as exc_info:
            Message(role="system", content="Invalid role")
        errors = exc_info.value.errors()
        assert any('role' in str(error) for error in errors)
    
    def test_missing_content(self):
        """Test validation error when content is missing."""
        with pytest.raises(ValidationError):
            Message(role="user")


class TestUserContext:
    """Tests for UserContext model."""
    
    def test_default_values(self):
        """Test UserContext with default values."""
        context = UserContext()
        assert context.chatHistory == []
        assert context.chatInterest is None
        assert context.userSummary == ""
        assert context.birthdate is None
        assert context.topics == []
    
    def test_full_context(self):
        """Test UserContext with all fields populated."""
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!")
        ]
        context = UserContext(
            chatHistory=messages,
            chatInterest="Python",
            userSummary="User is learning Python",
            birthdate="1990-01-01",
            topics=["Python", "AI", "Web Development"]
        )
        assert len(context.chatHistory) == 2
        assert context.chatHistory[0].role == "user"
        assert context.chatInterest == "Python"
        assert context.userSummary == "User is learning Python"
        assert context.birthdate == "1990-01-01"
        assert len(context.topics) == 3
    
    def test_partial_context(self):
        """Test UserContext with some fields populated."""
        context = UserContext(
            chatHistory=[Message(role="user", content="Test")],
            topics=["Technology"]
        )
        assert len(context.chatHistory) == 1
        assert context.chatInterest is None
        assert context.userSummary == ""
        assert len(context.topics) == 1
