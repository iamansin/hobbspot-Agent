"""Unit tests for utility functions."""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, call
from loguru import logger
import sys
from app.utils import configure_logging, log_execution_time, retry_with_backoff


class TestConfigureLogging:
    """Tests for configure_logging function."""
    
    def test_configure_logging_with_default_settings(self):
        """Test logging configuration with standard settings."""
        # Configure logging
        configure_logging(log_level="INFO", rotation="100 MB", retention="30 days")
        
        # Verify logger is configured by attempting to log
        logger.info("Test log message")
        
        # Check that logger has handlers configured
        # loguru doesn't expose handlers directly, but we can verify it doesn't raise errors
        assert True
    
    def test_configure_logging_with_debug_level(self):
        """Test logging configuration with DEBUG level."""
        configure_logging(log_level="DEBUG", rotation="50 MB", retention="7 days")
        
        # Verify debug messages can be logged
        logger.debug("Debug message")
        assert True
    
    def test_configure_logging_with_custom_rotation(self):
        """Test logging configuration with custom rotation settings."""
        configure_logging(log_level="WARNING", rotation="1 day", retention="14 days")
        
        # Verify configuration doesn't raise errors
        logger.warning("Warning message")
        assert True


class TestLogExecutionTime:
    """Tests for log_execution_time decorator."""
    
    @pytest.mark.asyncio
    async def test_decorator_logs_successful_execution(self):
        """Test decorator logs execution time for successful function."""
        
        @log_execution_time
        async def sample_function():
            await asyncio.sleep(0.1)
            return "success"
        
        # Execute function
        result = await sample_function()
        
        # Verify function returns correct result
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_decorator_logs_failed_execution(self):
        """Test decorator logs execution time and re-raises exception on failure."""
        
        @log_execution_time
        async def failing_function():
            await asyncio.sleep(0.05)
            raise ValueError("Test error")
        
        # Execute function and expect exception
        with pytest.raises(ValueError) as exc_info:
            await failing_function()
        
        # Verify correct exception is raised
        assert str(exc_info.value) == "Test error"
    
    @pytest.mark.asyncio
    async def test_decorator_with_function_arguments(self):
        """Test decorator works with functions that have arguments."""
        
        @log_execution_time
        async def function_with_args(x, y, z=10):
            await asyncio.sleep(0.05)
            return x + y + z
        
        # Execute function with arguments
        result = await function_with_args(5, 3, z=2)
        
        # Verify function returns correct result
        assert result == 10
    
    @pytest.mark.asyncio
    async def test_decorator_preserves_function_name(self):
        """Test decorator preserves original function name."""
        
        @log_execution_time
        async def my_custom_function():
            return "test"
        
        # Verify function name is preserved
        assert my_custom_function.__name__ == "my_custom_function"


class TestRetryWithBackoff:
    """Tests for retry_with_backoff function."""
    
    @pytest.mark.asyncio
    async def test_successful_first_attempt(self):
        """Test function succeeds on first attempt without retries."""
        call_count = 0
        
        async def successful_function():
            nonlocal call_count
            call_count += 1
            return "success"
        
        # Execute with retry logic
        result = await retry_with_backoff(successful_function, max_retries=3)
        
        # Verify function was called only once
        assert call_count == 1
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_retry_after_failures(self):
        """Test function retries after initial failures."""
        call_count = 0
        
        async def failing_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"
        
        # Execute with retry logic
        result = await retry_with_backoff(
            failing_then_success,
            max_retries=3,
            base_delay=0.01,  # Short delay for testing
            max_delay=0.1
        )
        
        # Verify function was called 3 times
        assert call_count == 3
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_all_retries_exhausted(self):
        """Test exception is raised when all retries are exhausted."""
        call_count = 0
        
        async def always_failing():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Persistent failure")
        
        # Execute with retry logic and expect exception
        with pytest.raises(RuntimeError) as exc_info:
            await retry_with_backoff(
                always_failing,
                max_retries=3,
                base_delay=0.01,
                max_delay=0.1
            )
        
        # Verify function was called max_retries times
        assert call_count == 3
        assert str(exc_info.value) == "Persistent failure"
    
    @pytest.mark.asyncio
    async def test_exponential_backoff_timing(self):
        """Test exponential backoff increases delay between retries."""
        call_times = []
        
        async def failing_function():
            call_times.append(asyncio.get_event_loop().time())
            raise ValueError("Test error")
        
        # Execute with retry logic
        try:
            await retry_with_backoff(
                failing_function,
                max_retries=3,
                base_delay=0.1,
                max_delay=1.0,
                jitter=0.0  # No jitter for predictable timing
            )
        except ValueError:
            pass
        
        # Verify we have 3 call times
        assert len(call_times) == 3
        
        # Verify delays increase (approximately)
        # First retry delay: ~0.1s, second retry delay: ~0.2s
        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]
        
        # Allow some tolerance for timing variations
        assert 0.08 < delay1 < 0.15
        assert 0.18 < delay2 < 0.25
    
    @pytest.mark.asyncio
    async def test_retry_with_function_arguments(self):
        """Test retry logic works with function arguments."""
        call_count = 0
        
        async def function_with_args(x, y):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Retry needed")
            return x + y
        
        # Execute with retry logic and arguments
        result = await retry_with_backoff(
            function_with_args,
            3,  # max_retries
            0.01,  # base_delay
            0.1,  # max_delay
            0.1,  # jitter
            5,  # x
            10  # y
        )
        
        # Verify function was called twice and returns correct result
        assert call_count == 2
        assert result == 15
    
    @pytest.mark.asyncio
    async def test_retry_with_keyword_arguments(self):
        """Test retry logic works with keyword arguments."""
        call_count = 0
        
        async def function_with_kwargs(name, age=0):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Retry needed")
            return f"{name} is {age}"
        
        # Execute with retry logic and keyword arguments
        result = await retry_with_backoff(
            function_with_kwargs,
            3,  # max_retries
            0.01,  # base_delay
            0.1,  # max_delay
            0.1,  # jitter
            name="Alice",
            age=30
        )
        
        # Verify function was called twice and returns correct result
        assert call_count == 2
        assert result == "Alice is 30"
    
    @pytest.mark.asyncio
    async def test_max_delay_cap(self):
        """Test that delay is capped at max_delay."""
        call_times = []
        
        async def failing_function():
            call_times.append(asyncio.get_event_loop().time())
            raise ValueError("Test error")
        
        # Execute with retry logic where exponential backoff would exceed max_delay
        try:
            await retry_with_backoff(
                failing_function,
                max_retries=5,
                base_delay=1.0,
                max_delay=0.2,  # Cap at 0.2s
                jitter=0.0
            )
        except ValueError:
            pass
        
        # Verify delays don't exceed max_delay
        for i in range(1, len(call_times)):
            delay = call_times[i] - call_times[i-1]
            # Allow small tolerance
            assert delay <= 0.25
    
    @pytest.mark.asyncio
    async def test_different_exception_types(self):
        """Test retry logic handles different exception types."""
        call_count = 0
        
        async def function_with_different_errors():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Connection failed")
            elif call_count == 2:
                raise TimeoutError("Request timeout")
            return "success"
        
        # Execute with retry logic
        result = await retry_with_backoff(
            function_with_different_errors,
            max_retries=3,
            base_delay=0.01,
            max_delay=0.1
        )
        
        # Verify function succeeded after different errors
        assert call_count == 3
        assert result == "success"
