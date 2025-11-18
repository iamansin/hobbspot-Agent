"""
Unit tests for cache manager.
"""
import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from app.cache import CacheManager
from app.models import UserContext, Message


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for cache testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def cache_manager(temp_cache_dir):
    """Create a CacheManager instance with temporary directory."""
    manager = CacheManager(cache_dir=temp_cache_dir, ttl=600)
    yield manager
    # Cleanup
    asyncio.run(manager.close())


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


class TestCacheManager:
    """Tests for CacheManager class."""
    
    @pytest.mark.asyncio
    async def test_cache_initialization(self, temp_cache_dir):
        """Test cache manager initialization."""
        manager = CacheManager(cache_dir=temp_cache_dir, ttl=300)
        assert manager.ttl == 300
        assert manager.cache is not None
        await manager.close()
    
    @pytest.mark.asyncio
    async def test_set_and_get_user_context(self, cache_manager, sample_user_context):
        """Test setting and getting user context from cache."""
        user_id = "user123"
        
        # Set context in cache
        await cache_manager.set(user_id, sample_user_context)
        
        # Get context from cache
        retrieved_context = await cache_manager.get(user_id)
        
        assert retrieved_context is not None
        assert retrieved_context.chatInterest == sample_user_context.chatInterest
        assert len(retrieved_context.chatHistory) == len(sample_user_context.chatHistory)
        assert retrieved_context.userSummary == sample_user_context.userSummary
        assert retrieved_context.birthdate == sample_user_context.birthdate
        assert retrieved_context.topics == sample_user_context.topics
    
    @pytest.mark.asyncio
    async def test_cache_miss(self, cache_manager):
        """Test cache miss for non-existent user."""
        user_id = "nonexistent_user"
        
        # Try to get non-existent user
        result = await cache_manager.get(user_id)
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_delete_user_context(self, cache_manager, sample_user_context):
        """Test deleting user context from cache."""
        user_id = "user456"
        
        # Set context
        await cache_manager.set(user_id, sample_user_context)
        
        # Verify it exists
        result = await cache_manager.get(user_id)
        assert result is not None
        
        # Delete context
        await cache_manager.delete(user_id)
        
        # Verify it's gone
        result = await cache_manager.get(user_id)
        assert result is None
    
    @pytest.mark.asyncio
    async def test_delete_nonexistent_user(self, cache_manager):
        """Test deleting non-existent user (should not raise error)."""
        user_id = "nonexistent_user"
        
        # Should not raise an error
        await cache_manager.delete(user_id)
    
    @pytest.mark.asyncio
    async def test_ttl_expiry(self, temp_cache_dir):
        """Test that cached data expires after TTL."""
        # Create cache manager with very short TTL
        short_ttl = 1  # 1 second
        manager = CacheManager(cache_dir=temp_cache_dir, ttl=short_ttl)
        
        user_id = "user_ttl_test"
        context = UserContext(
            chatHistory=[Message(role="user", content="Test")],
            chatInterest="Testing"
        )
        
        # Set context
        await manager.set(user_id, context)
        
        # Immediately retrieve - should exist
        result = await manager.get(user_id)
        assert result is not None
        
        # Wait for TTL to expire
        await asyncio.sleep(short_ttl + 0.5)
        
        # Try to retrieve - should be expired
        result = await manager.get(user_id)
        assert result is None
        
        await manager.close()
    
    @pytest.mark.asyncio
    async def test_check_and_summarize_not_needed(self, cache_manager, sample_user_context):
        """Test check_and_summarize when summarization is not needed."""
        user_id = "user789"
        max_messages = 10
        overlap = 5
        
        # Context has only 2 messages, threshold is 15 (10 + 5)
        needs_summarization, returned_context = await cache_manager.check_and_summarize(
            user_id, sample_user_context, max_messages, overlap
        )
        
        assert needs_summarization is False
        assert returned_context == sample_user_context
    
    @pytest.mark.asyncio
    async def test_check_and_summarize_needed(self, cache_manager):
        """Test check_and_summarize when summarization is needed."""
        user_id = "user_overflow"
        max_messages = 10
        overlap = 5
        
        # Create context with more messages than threshold
        messages = [
            Message(role="user" if i % 2 == 0 else "assistant", content=f"Message {i}")
            for i in range(20)  # 20 messages > threshold of 15
        ]
        context = UserContext(chatHistory=messages)
        
        needs_summarization, returned_context = await cache_manager.check_and_summarize(
            user_id, context, max_messages, overlap
        )
        
        assert needs_summarization is True
        assert returned_context == context
    
    @pytest.mark.asyncio
    async def test_check_and_summarize_exact_threshold(self, cache_manager):
        """Test check_and_summarize when message count equals threshold."""
        user_id = "user_exact"
        max_messages = 10
        overlap = 5
        threshold = max_messages + overlap  # 15
        
        # Create context with exactly threshold messages
        messages = [
            Message(role="user" if i % 2 == 0 else "assistant", content=f"Message {i}")
            for i in range(threshold)
        ]
        context = UserContext(chatHistory=messages)
        
        needs_summarization, returned_context = await cache_manager.check_and_summarize(
            user_id, context, max_messages, overlap
        )
        
        # At threshold, should not trigger (only > threshold triggers)
        assert needs_summarization is False
        assert returned_context == context
    
    @pytest.mark.asyncio
    async def test_corrupted_cache_data(self, cache_manager):
        """Test handling of corrupted cache data."""
        user_id = "user_corrupted"
        
        # Manually insert corrupted data into cache (invalid chatHistory structure)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: cache_manager.cache.set(
                user_id, 
                {"chatHistory": "not_a_list"},  # Invalid: should be a list
                expire=600
            )
        )
        
        # Try to get - should handle gracefully and return None
        result = await cache_manager.get(user_id)
        assert result is None
        
        # Verify corrupted entry was deleted
        result = await cache_manager.get(user_id)
        assert result is None
    
    @pytest.mark.asyncio
    async def test_multiple_users(self, cache_manager):
        """Test caching multiple users independently."""
        user1_id = "user1"
        user2_id = "user2"
        
        context1 = UserContext(
            chatHistory=[Message(role="user", content="User 1 message")],
            chatInterest="Topic 1"
        )
        context2 = UserContext(
            chatHistory=[Message(role="user", content="User 2 message")],
            chatInterest="Topic 2"
        )
        
        # Set both contexts
        await cache_manager.set(user1_id, context1)
        await cache_manager.set(user2_id, context2)
        
        # Retrieve and verify both
        retrieved1 = await cache_manager.get(user1_id)
        retrieved2 = await cache_manager.get(user2_id)
        
        assert retrieved1.chatInterest == "Topic 1"
        assert retrieved2.chatInterest == "Topic 2"
        assert retrieved1.chatHistory[0].content == "User 1 message"
        assert retrieved2.chatHistory[0].content == "User 2 message"
    
    @pytest.mark.asyncio
    async def test_update_existing_context(self, cache_manager, sample_user_context):
        """Test updating an existing cached context."""
        user_id = "user_update"
        
        # Set initial context
        await cache_manager.set(user_id, sample_user_context)
        
        # Update context with new message
        updated_context = UserContext(
            chatHistory=sample_user_context.chatHistory + [
                Message(role="user", content="New message")
            ],
            chatInterest=sample_user_context.chatInterest,
            userSummary=sample_user_context.userSummary,
            birthdate=sample_user_context.birthdate,
            topics=sample_user_context.topics
        )
        
        # Set updated context
        await cache_manager.set(user_id, updated_context)
        
        # Retrieve and verify
        retrieved = await cache_manager.get(user_id)
        assert len(retrieved.chatHistory) == 3
        assert retrieved.chatHistory[-1].content == "New message"
