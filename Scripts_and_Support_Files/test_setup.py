#!/usr/bin/env python3
"""
Test script to verify the sentiment analysis setup.
Run this before using the main script to ensure everything is configured correctly.
"""

import os
import json
import pandas as pd
from pathlib import Path

def test_file_exists(filepath: str, description: str) -> bool:
    """Test if a file exists and is readable."""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} - NOT FOUND")
        return False

def test_csv_structure(filepath: str, description: str) -> bool:
    """Test if a CSV file has the expected structure."""
    try:
        df = pd.read_csv(filepath)
        print(f"✅ {description}: {filepath}")
        print(f"   - Rows: {len(df)}")
        print(f"   - Columns: {list(df.columns)}")
        return True
    except Exception as e:
        print(f"❌ {description}: {filepath} - ERROR: {e}")
        return False

def test_json_structure(filepath: str, description: str) -> bool:
    """Test if a JSON file has the expected structure."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ {description}: {filepath}")
        print(f"   - Keys: {list(data.keys())}")
        return True
    except Exception as e:
        print(f"❌ {description}: {filepath} - ERROR: {e}")
        return False

def main():
    """Run all tests."""
    print("🔍 Testing Sentiment Analysis Setup\n")
    
    # Test required files
    required_files = [
        ("MarketcastOpensOnly.csv", "Main survey data"),
        ("HumanClassification.csv", "Human classification examples"),
        ("sentiment_map.json", "Sentiment mapping definitions")
    ]
    
    all_files_ok = True
    for filepath, description in required_files:
        if not test_file_exists(filepath, description):
            all_files_ok = False
    
    print("\n📊 Testing File Structures\n")
    
    # Test CSV structures
    if test_csv_structure("MarketcastOpensOnly.csv", "Main survey data structure"):
        df = pd.read_csv("MarketcastOpensOnly.csv")
        print(f"   - Sample UUID: {df.iloc[0]['uuid'] if 'uuid' in df.columns else 'No UUID column'}")
        print(f"   - Sample OETopic: {df.iloc[0]['OETopic'][:50] if 'OETopic' in df.columns else 'No OETopic column'}...")
    else:
        all_files_ok = False
    
    if test_csv_structure("HumanClassification.csv", "Human classification structure"):
        df = pd.read_csv("HumanClassification.csv")
        print(f"   - Sample UUID: {df.iloc[0]['uuid'] if 'uuid' in df.columns else 'No UUID column'}")
    else:
        all_files_ok = False
    
    # Test JSON structure
    if test_json_structure("sentiment_map.json", "Sentiment mapping structure"):
        with open("sentiment_map.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"   - Sentiment codes: {list(data.keys())}")
        for code, details in data.items():
            print(f"     - Code {code}: {details.get('name', 'No name')}")
    else:
        all_files_ok = False
    
    # Test Python dependencies
    print("\n🐍 Testing Python Dependencies\n")
    
    try:
        import pandas
        print(f"✅ pandas: {pandas.__version__}")
    except ImportError:
        print("❌ pandas: NOT INSTALLED")
        all_files_ok = False
    
    try:
        import openai
        print(f"✅ openai: {openai.__version__}")
    except ImportError:
        print("❌ openai: NOT INSTALLED")
        all_files_ok = False
    
    # Test environment variables
    print("\n🔑 Testing Environment Variables\n")
    
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
        print(f"✅ OPENAI_API_KEY: {masked_key}")
    else:
        print("❌ OPENAI_API_KEY: NOT SET")
        print("   Set this environment variable or pass --api-key when running the script")
        all_files_ok = False
    
    # Summary
    print("\n" + "="*50)
    if all_files_ok:
        print("🎉 All tests passed! Your setup is ready.")
        print("\nNext steps:")
        print("1. Ensure your OpenAI API key is set")
        print("2. Run: python ImpactAPIcalls.py --start-row 1 --end-row 10")
        print("3. Check the output files and logs")
    else:
        print("⚠️  Some tests failed. Please fix the issues above before proceeding.")
        print("\nCommon fixes:")
        print("- Install missing packages: pip install -r requirements.txt")
        print("- Set your OpenAI API key: export OPENAI_API_KEY='your-key'")
        print("- Ensure all required files are in the same directory")
    
    print("="*50)

if __name__ == "__main__":
    main()
