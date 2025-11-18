"""Utility functions for logging, timing, and retry logic."""

from loguru import logger
import sys
import asyncio
from functools import wraps
from typing import Callable, Any, TypeVar
import random


T = TypeVar('T')


def configure_logging(log_level: str, rotation: str, retention: str) -> None:
    """
    Configure loguru logging with stdout and file handlers.
    
    Sets up structured logging with:
    - Console output with colored formatting
    - File output with rotation and retention
    - Configurable log levels
    
    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        rotation: When to rotate log files (e.g., "100 MB", "1 day")
        retention: How long to keep old log files (e.g., "30 days", "1 week")
    
    Requirements: 13.1, 13.2, 13.12
    """
    # Remove default handler
    logger.remove()
    
    # Add stdout handler with colored format
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        colorize=True
    )
    
    # Add file handler with rotation and retention
    logger.add(
        "logs/chat_agent_{time:YYYY-MM-DD}.log",
        rotation=rotation,
        retention=retention,
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        enqueue=True,  # Thread-safe logging
        backtrace=True,  # Include full stack trace on errors
        diagnose=True  # Include variable values in stack traces
    )
    
    logger.info(f"Logging configured: level={log_level}, rotation={rotation}, retention={retention}")


def log_execution_time(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to log function execution time.
    
    Logs the duration of async function execution and handles exceptions.
    Useful for monitoring performance of I/O operations.
    
    Args:
        func: Async function to wrap
        
    Returns:
        Wrapped function that logs execution time
        
    Requirements: 13.13
    """
    @wraps(func)
    async def wrapper(*args, **kwargs) -> T:
        start = asyncio.get_event_loop().time()
        func_name = func.__name__
        
        try:
            result = await func(*args, **kwargs)
            duration = asyncio.get_event_loop().time() - start
            logger.info(f"{func_name} completed in {duration:.3f}s")
            return result
        except Exception as e:
            duration = asyncio.get_event_loop().time() - start
            logger.error(f"{func_name} failed after {duration:.3f}s: {e}")
            raise
    
    return wrapper


async def retry_with_backoff(
    func: Callable[..., T],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    jitter: float = 0.1,
    operation_name: str = None,
    *args,
    **kwargs
) -> T:
    """
    Retry async function with exponential backoff.
    
    Implements exponential backoff with jitter for retrying failed operations.
    Useful for handling transient failures in external API calls.
    
    Args:
        func: Async function to retry
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Initial delay in seconds (default: 1.0)
        max_delay: Maximum delay in seconds (default: 10.0)
        jitter: Random jitter factor (default: 0.1 = 10%)
        operation_name: Optional descriptive name for the operation (for better logging)
        *args: Positional arguments to pass to func
        **kwargs: Keyword arguments to pass to func
        
    Returns:
        Result from successful function execution
        
    Raises:
        Exception: The last exception if all retries fail
        
    Requirements: 13.11
    """
    import traceback
    
    func_name = operation_name or func.__name__
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            result = await func(*args, **kwargs)
            if attempt > 0:
                logger.info(f"[{func_name}] succeeded on attempt {attempt + 1}")
            return result
        except Exception as e:
            last_exception = e
            
            # Extract comprehensive error details
            error_msg = str(e)
            error_type = type(e).__name__
            
            # Get full traceback for detailed debugging
            tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
            full_traceback = ''.join(tb_lines)
            
            # Log detailed error information
            if attempt < max_retries - 1:
                # Calculate delay with exponential backoff
                delay = min(base_delay * (2 ** attempt), max_delay)
                
                # Add random jitter
                jitter_amount = delay * jitter * random.uniform(-1, 1)
                delay_with_jitter = max(0, delay + jitter_amount)
                
                logger.warning(
                    f"[{func_name}] {error_type} on attempt {attempt + 1}/{max_retries}\n"
                    f"Error message: {error_msg}\n"
                    f"Error type: {error_type}\n"
                    f"Retrying in {delay_with_jitter:.2f}s...\n"
                    f"Traceback:\n{full_traceback}"
                )
                
                await asyncio.sleep(delay_with_jitter)
            else:
                logger.error(
                    f"[{func_name}] {error_type} after {max_retries} attempts - GIVING UP\n"
                    f"Error message: {error_msg}\n"
                    f"Error type: {error_type}\n"
                    f"Full traceback:\n{full_traceback}"
                )
    
    # All retries exhausted
    raise last_exception
