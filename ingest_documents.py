import os
import json
import openai
from elasticsearch import Elasticsearch
from openai import AzureOpenAI
from dotenv import load_dotenv
load_dotenv()


# Ingest documents


# Sample documents
documents = [
    {"id": "1", "content": "LangGraph is a powerful orchestration framework for multi-agent systems."},
    {"id": "2", "content": "Elasticsearch is used for storing and retrieving vector embeddings efficiently."}
]

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

def get_embedding(text):
    response = client.embeddings.create(
        input=[text],
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT")
    )
    return response.data[0].embedding

# Connect to Elasticsearch
es = Elasticsearch("http://localhost:9200")
index_name = "rag_index"


for doc in documents:
    embedding = get_embedding(doc["content"])
    es.index(index=index_name, id=doc["id"], body={
        "content": doc["content"],
        "embedding": embedding
    })

print("✅ Documents ingested successfully.")

# Load environment variables
load_dotenv()

openai.api_type = "azure"
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION")