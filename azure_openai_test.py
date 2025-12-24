"""
Azure OpenAI Connection Test Script
This script tests the connection to Azure OpenAI service using environment variables.
"""

import os
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_azure_openai_connection():
    """Test Azure OpenAI connection with environment variables."""
    
    # Get configuration from environment variables
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4")
    
    if not endpoint or not api_key:
        print("❌ Missing required environment variables:")
        print("   - AZURE_OPENAI_ENDPOINT")
        print("   - AZURE_OPENAI_API_KEY")
        print("\nPlease set these variables in your .env file")
        return False
    
    try:
        # Create Azure OpenAI client
        client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=api_key,
        )
        
        # Test with a simple completion
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": "Hello! This is a test message."
                }
            ],
            model=deployment,
            max_tokens=50
        )
        
        print("✅ Azure OpenAI connection successful!")
        print(f"📍 Endpoint: {endpoint}")
        print(f"🤖 Model: {deployment}")
        print(f"📝 Response: {response.choices[0].message.content.strip()}")
        return True
        
    except Exception as e:
        print(f"❌ Azure OpenAI connection failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_azure_openai_connection()