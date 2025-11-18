"""
Automated E2E test runner script.

This script:
1. Checks if .env file exists
2. Validates environment setup
3. Starts the FastAPI application in the background
4. Runs the E2E tests
5. Stops the application

Requirements: 14.2, 14.6, 14.7, 14.12
"""

import subprocess
import sys
import os
import time
import signal
import httpx
import asyncio
from pathlib import Path


def check_env_file():
    """Check if .env file exists and is configured."""
    if not os.path.exists('.env'):
        print("❌ .env file not found!")
        print("\nPlease create a .env file with the following variables:")
        print("  - APPWRITE_ENDPOINT")
        print("  - APPWRITE_PROJECT_ID")
        print("  - APPWRITE_API_KEY")
        print("  - APPWRITE_DATABASE_ID")
        print("  - APPWRITE_COLLECTION_ID")
        print("  - OPENAI_API_KEY")
        print("  - GEMINI_API_KEY")
        print("  - BRAVE_API_KEY")
        print("\nYou can copy .env.template to .env and fill in your values:")
        print("  copy .env.template .env")
        return False
    
    print("✓ .env file found")
    return True


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import fastapi
        import uvicorn
        import httpx
        import diskcache
        import appwrite
        import openai
        import google.generativeai
        import loguru
        import pydantic
        import slowapi
        
        print("✓ All dependencies installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\nPlease install dependencies:")
        print("  pip install -r requirements.txt")
        return False


async def wait_for_server(url: str, timeout: int = 30):
    """Wait for the server to be ready."""
    print(f"Waiting for server at {url}...")
    
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{url}/health", timeout=5.0)
                if response.status_code == 200:
                    print("✓ Server is ready")
                    return True
        except:
            pass
        
        await asyncio.sleep(1)
    
    print("❌ Server failed to start within timeout")
    return False


def start_server():
    """Start the FastAPI server in the background."""
    print("Starting FastAPI server...")
    
    # Start uvicorn in a subprocess
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    return process


def stop_server(process):
    """Stop the FastAPI server."""
    print("\nStopping server...")
    
    if process:
        try:
            process.terminate()
            process.wait(timeout=5)
            print("✓ Server stopped")
        except:
            process.kill()
            print("✓ Server killed")


async def run_tests():
    """Run the E2E tests."""
    print("\n" + "="*80)
    print("RUNNING E2E TESTS")
    print("="*80 + "\n")
    
    # Import and run the test suite
    sys.path.insert(0, 'tests')
    from test_e2e_manual import run_all_tests
    
    await run_all_tests()


async def main():
    """Main execution function."""
    print("="*80)
    print("CHAT AGENT E2E TEST RUNNER")
    print("="*80 + "\n")
    
    # Step 1: Check environment
    if not check_env_file():
        return 1
    
    if not check_dependencies():
        return 1
    
    # Step 2: Start server
    server_process = None
    
    try:
        server_process = start_server()
        
        # Wait for server to be ready
        if not await wait_for_server("http://127.0.0.1:8000"):
            return 1
        
        # Step 3: Run tests
        await run_tests()
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠ Tests interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        # Step 4: Cleanup
        if server_process:
            stop_server(server_process)


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
