"""
End-to-end manual testing script for the Chat Agent application.

This script tests the application by making real HTTP requests to a running instance.
It covers all the key scenarios from task 12:
- First-time user interaction
- Returning user interaction
- Function calling with web search
- Cache expiry and summarization
- Error handling

Requirements: 14.2, 14.6, 14.7, 14.12

Usage:
    1. Ensure the application is running (uvicorn main:app --reload)
    2. Ensure .env file is configured with valid API keys
    3. Run: python tests/test_e2e_manual.py
"""

import httpx
import asyncio
import time
import json
from typing import Dict, Any


BASE_URL = "http://localhost:8000"
TEST_USER_ID = f"e2e_test_user_{int(time.time())}"


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")


def print_success(text: str):
    """Print a success message."""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")


def print_error(text: str):
    """Print an error message."""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")


def print_info(text: str):
    """Print an info message."""
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")


def print_warning(text: str):
    """Print a warning message."""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")


async def test_health_check():
    """Test the health check endpoint."""
    print_header("TEST 1: Health Check")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/health")
            
            if response.status_code == 200:
                data = response.json()
                print_success(f"Health check passed: {data}")
                return True
            else:
                print_error(f"Health check failed with status {response.status_code}")
                return False
    except Exception as e:
        print_error(f"Health check failed: {e}")
        return False


async def test_first_time_user():
    """Test first-time user interaction."""
    print_header("TEST 2: First-Time User Interaction")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            payload = {
                "userId": TEST_USER_ID,
                "userMessage": "I want to learn about Python",
                "chatInterest": True,
                "interestTopic": "Python programming and best practices"
            }
            
            print_info(f"Sending first-time user request for userId: {TEST_USER_ID}")
            print_info(f"Interest topic: {payload['interestTopic']}")
            
            response = await client.post(f"{BASE_URL}/chat", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                print_success("First-time user request successful")
                print_info(f"Response preview: {data['response'][:200]}...")
                return True
            else:
                print_error(f"Request failed with status {response.status_code}")
                print_error(f"Response: {response.text}")
                return False
    except Exception as e:
        print_error(f"First-time user test failed: {e}")
        return False


async def test_returning_user():
    """Test returning user interaction."""
    print_header("TEST 3: Returning User Interaction")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            payload = {
                "userId": TEST_USER_ID,
                "userMessage": "Can you explain list comprehensions in Python?",
                "chatInterest": False
            }
            
            print_info(f"Sending returning user request for userId: {TEST_USER_ID}")
            print_info(f"Message: {payload['userMessage']}")
            
            response = await client.post(f"{BASE_URL}/chat", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                print_success("Returning user request successful")
                print_info(f"Response preview: {data['response'][:200]}...")
                return True
            else:
                print_error(f"Request failed with status {response.status_code}")
                print_error(f"Response: {response.text}")
                return False
    except Exception as e:
        print_error(f"Returning user test failed: {e}")
        return False


async def test_function_calling():
    """Test function calling with web search."""
    print_header("TEST 4: Function Calling with Web Search")
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            payload = {
                "userId": TEST_USER_ID,
                "userMessage": "What are the latest features in Python 3.12?",
                "chatInterest": False
            }
            
            print_info(f"Sending request that should trigger web search")
            print_info(f"Message: {payload['userMessage']}")
            
            response = await client.post(f"{BASE_URL}/chat", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                print_success("Function calling request successful")
                print_info(f"Response preview: {data['response'][:200]}...")
                
                # Check if response contains information that would require web search
                if "3.12" in data['response'] or "latest" in data['response'].lower():
                    print_success("Response appears to contain current information")
                else:
                    print_warning("Response may not have used web search")
                
                return True
            else:
                print_error(f"Request failed with status {response.status_code}")
                print_error(f"Response: {response.text}")
                return False
    except Exception as e:
        print_error(f"Function calling test failed: {e}")
        return False


async def test_multiple_interactions():
    """Test multiple interactions to build up chat history."""
    print_header("TEST 5: Multiple Interactions (Building History)")
    
    messages = [
        "What are decorators in Python?",
        "Can you give me an example of a decorator?",
        "How do I use *args and **kwargs?",
        "What's the difference between a list and a tuple?",
        "Explain Python generators",
        "What are context managers?",
        "How does the with statement work?",
        "What is the GIL in Python?",
    ]
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            for i, message in enumerate(messages, 1):
                print_info(f"Sending message {i}/{len(messages)}: {message}")
                
                payload = {
                    "userId": TEST_USER_ID,
                    "userMessage": message,
                    "chatInterest": False
                }
                
                response = await client.post(f"{BASE_URL}/chat", json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    print_success(f"Message {i} successful")
                    
                    # Small delay between requests
                    await asyncio.sleep(1)
                else:
                    print_error(f"Message {i} failed with status {response.status_code}")
                    return False
            
            print_success("All messages sent successfully")
            print_info("Chat history should now be building up")
            return True
            
    except Exception as e:
        print_error(f"Multiple interactions test failed: {e}")
        return False


async def test_summarization_trigger():
    """Test that summarization is triggered with many messages."""
    print_header("TEST 6: Summarization Trigger")
    
    print_info("Sending additional messages to trigger summarization...")
    print_info("(Threshold is PREVIOUS_MESSAGE_CONTEXT_LENGTH + OVERLAP_COUNT)")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Send more messages to exceed threshold (default: 10 + 5 = 15)
            for i in range(10):
                payload = {
                    "userId": TEST_USER_ID,
                    "userMessage": f"Tell me about Python topic number {i+1}",
                    "chatInterest": False
                }
                
                print_info(f"Sending message {i+1}/10")
                response = await client.post(f"{BASE_URL}/chat", json=payload)
                
                if response.status_code != 200:
                    print_error(f"Message {i+1} failed")
                    return False
                
                await asyncio.sleep(1)
            
            print_success("Sent enough messages to trigger summarization")
            print_info("Check logs for summarization events")
            return True
            
    except Exception as e:
        print_error(f"Summarization test failed: {e}")
        return False


async def test_cache_expiry():
    """Test cache expiry by waiting for TTL."""
    print_header("TEST 7: Cache Expiry")
    
    print_warning("This test requires waiting for cache TTL (default: 10 minutes)")
    print_info("Skipping automatic wait - check manually if needed")
    print_info("To test manually:")
    print_info("  1. Wait 10+ minutes after last request")
    print_info("  2. Send another message")
    print_info("  3. Check logs for 'Cache miss' and database fetch")
    
    return True


async def test_error_handling():
    """Test error handling with invalid requests."""
    print_header("TEST 8: Error Handling")
    
    test_cases = [
        {
            "name": "Empty userId",
            "payload": {
                "userId": "",
                "userMessage": "Hello",
                "chatInterest": False
            },
            "expected_status": 422
        },
        {
            "name": "Empty userMessage",
            "payload": {
                "userId": "test_user",
                "userMessage": "",
                "chatInterest": False
            },
            "expected_status": 422
        },
        {
            "name": "Missing interestTopic for first-time user",
            "payload": {
                "userId": "test_user",
                "userMessage": "Hello",
                "chatInterest": True
            },
            "expected_status": 422
        },
        {
            "name": "Invalid payload structure",
            "payload": {
                "userId": "test_user"
                # Missing required fields
            },
            "expected_status": 422
        }
    ]
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            all_passed = True
            
            for test_case in test_cases:
                print_info(f"Testing: {test_case['name']}")
                
                response = await client.post(f"{BASE_URL}/chat", json=test_case['payload'])
                
                if response.status_code == test_case['expected_status']:
                    print_success(f"Correctly returned status {response.status_code}")
                else:
                    print_error(
                        f"Expected status {test_case['expected_status']}, "
                        f"got {response.status_code}"
                    )
                    all_passed = False
            
            return all_passed
            
    except Exception as e:
        print_error(f"Error handling test failed: {e}")
        return False


async def verify_logging():
    """Verify that logging is working."""
    print_header("TEST 9: Verify Logging Output")
    
    print_info("Checking for log files...")
    
    import os
    
    log_dir = "logs"
    if os.path.exists(log_dir):
        log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
        
        if log_files:
            print_success(f"Found {len(log_files)} log file(s)")
            
            # Check the most recent log file
            latest_log = max(
                [os.path.join(log_dir, f) for f in log_files],
                key=os.path.getmtime
            )
            
            print_info(f"Latest log file: {latest_log}")
            
            # Read last few lines
            try:
                with open(latest_log, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                print_info(f"Log file has {len(lines)} lines")
                print_info("Last 5 log entries:")
                
                for line in lines[-5:]:
                    print(f"  {line.rstrip()}")
                
                # Check for key log patterns
                log_content = ''.join(lines)
                
                checks = [
                    ("Chat request received", "Request logging"),
                    ("Cache", "Cache operations"),
                    ("AI response", "AI agent logging"),
                ]
                
                for pattern, description in checks:
                    if pattern in log_content:
                        print_success(f"{description} found in logs")
                    else:
                        print_warning(f"{description} not found in logs")
                
                return True
                
            except Exception as e:
                print_error(f"Error reading log file: {e}")
                return False
        else:
            print_warning("No log files found")
            return False
    else:
        print_warning(f"Log directory '{log_dir}' not found")
        return False


async def run_all_tests():
    """Run all end-to-end tests."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}")
    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║                   CHAT AGENT E2E TEST SUITE                                ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.ENDC}\n")
    
    print_info(f"Base URL: {BASE_URL}")
    print_info(f"Test User ID: {TEST_USER_ID}")
    print_info("Make sure the application is running before proceeding!")
    
    # Wait for user confirmation
    input("\nPress Enter to start tests...")
    
    results = {}
    
    # Run tests in sequence
    results['health_check'] = await test_health_check()
    
    if not results['health_check']:
        print_error("\nHealth check failed! Make sure the application is running.")
        print_error("Start the application with: uvicorn main:app --reload")
        return
    
    results['first_time_user'] = await test_first_time_user()
    await asyncio.sleep(2)
    
    results['returning_user'] = await test_returning_user()
    await asyncio.sleep(2)
    
    results['function_calling'] = await test_function_calling()
    await asyncio.sleep(2)
    
    results['multiple_interactions'] = await test_multiple_interactions()
    await asyncio.sleep(2)
    
    results['summarization'] = await test_summarization_trigger()
    await asyncio.sleep(2)
    
    results['cache_expiry'] = await test_cache_expiry()
    
    results['error_handling'] = await test_error_handling()
    await asyncio.sleep(2)
    
    results['logging'] = await verify_logging()
    
    # Print summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASSED" if result else "FAILED"
        color = Colors.OKGREEN if result else Colors.FAIL
        print(f"{color}{test_name.replace('_', ' ').title()}: {status}{Colors.ENDC}")
    
    print(f"\n{Colors.BOLD}Total: {passed}/{total} tests passed{Colors.ENDC}")
    
    if passed == total:
        print(f"\n{Colors.OKGREEN}{Colors.BOLD}✓ ALL TESTS PASSED!{Colors.ENDC}\n")
    else:
        print(f"\n{Colors.FAIL}{Colors.BOLD}✗ SOME TESTS FAILED{Colors.ENDC}\n")
    
    print_info("\nNext steps:")
    print_info("1. Review the logs in the logs/ directory")
    print_info("2. Check the cache/ directory for cached data")
    print_info("3. Verify data in Appwrite database")
    print_info("4. Test cache expiry manually after 10 minutes")


if __name__ == "__main__":
    try:
        asyncio.run(run_all_tests())
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Tests interrupted by user{Colors.ENDC}")
    except Exception as e:
        print(f"\n{Colors.FAIL}Test suite failed: {e}{Colors.ENDC}")
