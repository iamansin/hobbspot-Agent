"""Cache manager using DiskCache for user context storage."""

from diskcache import Cache
from typing import Optional
from loguru import logger
import asyncio
from app.models import UserContext, Message


class CacheManager:
    """
    Manages user context caching with TTL support using DiskCache.
    
    Provides async wrapper around DiskCache operations with comprehensive
    logging for cache hits, misses, and TTL expiry events.
    
    Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 13.4, 11.2
    """
    
    def __init__(self, cache_dir: str, ttl: int):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory path for DiskCache storage
            ttl: Time-to-live in seconds for cached entries
        """
        self.cache = Cache(cache_dir)
        self.ttl = ttl
        logger.info(f"CacheManager initialized: directory={cache_dir}, ttl={ttl}s")
    
    async def get(self, user_id: str) -> Optional[UserContext]:
        """
        Get user context from cache.
        
        Retrieves user context if present and not expired. Logs cache hits,
        misses, and TTL expiry events.
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            UserContext if found and not expired, None otherwise
            
        Requirements: 2.2, 13.4
        """
        # Run blocking cache operation in thread pool
        loop = asyncio.get_event_loop()
        cache_data = await loop.run_in_executor(None, self.cache.get, user_id)
        
        if cache_data is None:
            logger.info(f"Cache miss for user_id={user_id}")
            return None
        
        # Check if data has expired (DiskCache handles TTL internally, but we log it)
        logger.info(f"Cache hit for user_id={user_id}")
        
        try:
            # Convert dict back to UserContext
            user_context = UserContext(**cache_data)
            return user_context
        except Exception as e:
            logger.error(f"Failed to deserialize cached data for user_id={user_id}: {e}")
            # Delete corrupted cache entry
            await self.delete(user_id)
            return None
    
    async def set(self, user_id: str, context: UserContext) -> None:
        """
        Set user context in cache with TTL.
        
        Stores user context with configured TTL. Logs the operation.
        
        Args:
            user_id: Unique user identifier
            context: User context to cache
            
        Requirements: 2.4, 13.4
        """
        # Convert UserContext to dict for storage
        cache_data = context.model_dump()
        
        # Run blocking cache operation in thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self.cache.set(user_id, cache_data, expire=self.ttl)
        )
        
        logger.info(f"Cache set for user_id={user_id}, ttl={self.ttl}s")
    
    async def delete(self, user_id: str) -> None:
        """
        Remove user from cache.
        
        Deletes cached user context. Logs the operation.
        
        Args:
            user_id: Unique user identifier
            
        Requirements: 2.3, 13.4
        """
        # Run blocking cache operation in thread pool
        loop = asyncio.get_event_loop()
        deleted = await loop.run_in_executor(None, self.cache.delete, user_id)
        
        if deleted:
            logger.info(f"Cache deleted for user_id={user_id}")
        else:
            logger.debug(f"Cache delete called for non-existent user_id={user_id}")
    
    async def check_and_summarize(
        self,
        user_id: str,
        context: UserContext,
        max_messages: int,
        overlap: int
    ) -> tuple[bool, UserContext]:
        """
        Check if summarization is needed and return updated context.
        
        Determines if chat history exceeds the threshold and needs summarization.
        Does not perform the actual summarization (that's done by AI agent),
        but detects when it's needed.
        
        Args:
            user_id: Unique user identifier
            context: Current user context
            max_messages: Maximum number of messages to retain (PREVIOUS_MESSAGE_CONTEXT_LENGTH)
            overlap: Overlap count for summarization threshold
            
        Returns:
            Tuple of (needs_summarization, context)
            - needs_summarization: True if summarization should be triggered
            - context: The same context passed in (unchanged)
            
        Requirements: 2.5, 11.2
        """
        current_message_count = len(context.chatHistory)
        threshold = max_messages + overlap
        
        if current_message_count > threshold:
            logger.info(
                f"Summarization needed for user_id={user_id}: "
                f"message_count={current_message_count}, threshold={threshold}"
            )
            return True, context
        else:
            logger.debug(
                f"No summarization needed for user_id={user_id}: "
                f"message_count={current_message_count}, threshold={threshold}"
            )
            return False, context
    
    async def cleanup_expired(self) -> int:
        """
        Remove expired entries from cache.
        
        DiskCache uses lazy deletion, so expired entries remain on disk
        until accessed. This method forces cleanup of expired entries.
        
        Returns:
            Number of expired entries removed
            
        Requirements: 2.3, 11.2
        """
        loop = asyncio.get_event_loop()
        count = await loop.run_in_executor(None, self.cache.expire)
        logger.info(f"Cache cleanup: removed {count} expired entries")
        return count
    
    async def close(self) -> None:
        """
        Close the cache and cleanup resources.
        
        Should be called during application shutdown.
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.cache.close)
        logger.info("CacheManager closed")
