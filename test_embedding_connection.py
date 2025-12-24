import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()

azure_embeddings = AzureOpenAIEmbeddings(
     azure_deployment="text-embedding-ada-002",
     model="text-embedding-ada-002",
     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
     openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
     openai_api_key=os.getenv("AZURE_OPENAI_API_KEY")
 )
print(azure_embeddings.embed_query("hello world"))