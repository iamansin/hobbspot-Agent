"""
Setup verification script for E2E testing.

This script checks if the environment is properly configured for running E2E tests.
"""

import os
import sys
from pathlib import Path


def check_env_file():
    """Check if .env file exists and has required variables."""
    print("Checking .env file...")
    
    if not os.path.exists('.env'):
        print("❌ .env file not found!")
        print("\n📝 To create .env file:")
        print("   1. Copy .env.template to .env:")
        print("      copy .env.template .env")
        print("   2. Edit .env and add your API keys")
        return False
    
    print("✓ .env file exists")
    
    # Check for required variables
    required_vars = [
        'APPWRITE_ENDPOINT',
        'APPWRITE_PROJECT_ID',
        'APPWRITE_API_KEY',
        'APPWRITE_DATABASE_ID',
        'APPWRITE_COLLECTION_ID',
        'OPENAI_API_KEY',
        'GEMINI_API_KEY',
        'BRAVE_API_KEY'
    ]
    
    missing_vars = []
    
    with open('.env', 'r') as f:
        content = f.read()
        
        for var in required_vars:
            if f"{var}=" not in content or f"{var}=your_" in content:
                missing_vars.append(var)
    
    if missing_vars:
        print(f"⚠ Warning: The following variables may not be configured:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\n📝 Please update these in your .env file")
        return False
    
    print("✓ All required environment variables appear to be set")
    return True


def check_dependencies():
    """Check if required Python packages are installed."""
    print("\nChecking dependencies...")
    
    required_packages = [
        ('fastapi', 'FastAPI'),
        ('uvicorn', 'Uvicorn'),
        ('httpx', 'HTTPX'),
        ('diskcache', 'DiskCache'),
        ('appwrite', 'Appwrite SDK'),
        ('openai', 'OpenAI SDK'),
        ('google.generativeai', 'Google Gemini SDK'),
        ('loguru', 'Loguru'),
        ('pydantic', 'Pydantic'),
        ('slowapi', 'SlowAPI'),
        ('pytest', 'Pytest')
    ]
    
    missing = []
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"❌ {name}")
            missing.append(name)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("\n📝 To install missing packages:")
        print("   pip install -r requirements.txt")
        return False
    
    print("\n✓ All dependencies installed")
    return True


def check_directories():
    """Check if required directories exist."""
    print("\nChecking directories...")
    
    dirs = ['cache', 'logs', 'app', 'tests']
    
    for dir_name in dirs:
        if os.path.exists(dir_name):
            print(f"✓ {dir_name}/")
        else:
            print(f"⚠ {dir_name}/ not found")
    
    return True


def check_main_files():
    """Check if main application files exist."""
    print("\nChecking application files...")
    
    files = [
        'main.py',
        'app/config.py',
        'app/models.py',
        'app/cache.py',
        'app/db_service.py',
        'app/ai_agent.py',
        'app/search.py',
        'app/utils.py'
    ]
    
    all_exist = True
    
    for file in files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"❌ {file}")
            all_exist = False
    
    if not all_exist:
        print("\n❌ Some application files are missing!")
        return False
    
    print("\n✓ All application files present")
    return True


def main():
    """Run all checks."""
    print("="*80)
    print("CHAT AGENT E2E TEST SETUP VERIFICATION")
    print("="*80 + "\n")
    
    checks = [
        ("Environment Configuration", check_env_file),
        ("Python Dependencies", check_dependencies),
        ("Directory Structure", check_directories),
        ("Application Files", check_main_files)
    ]
    
    results = []
    
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Error during {name} check: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80 + "\n")
    
    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
        if not result:
            all_passed = False
    
    print("\n" + "="*80)
    
    if all_passed:
        print("\n✓ Setup verification complete! You're ready to run E2E tests.")
        print("\n📝 Next steps:")
        print("   1. Start the server: uvicorn main:app --reload")
        print("   2. Run E2E tests: python tests/test_e2e_manual.py")
        print("   OR")
        print("   Run automated: python run_e2e_tests.py")
        return 0
    else:
        print("\n❌ Setup verification failed. Please fix the issues above.")
        print("\n📝 Common fixes:")
        print("   - Create .env file: copy .env.template .env")
        print("   - Install dependencies: pip install -r requirements.txt")
        print("   - Ensure you're in the project root directory")
        return 1


if __name__ == "__main__":
    sys.exit(main())
