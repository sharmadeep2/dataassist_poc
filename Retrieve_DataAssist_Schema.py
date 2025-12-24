import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from openai import AzureOpenAI

# Load environment variables
load_dotenv()

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

# Connect to Elasticsearch
es = Elasticsearch("http://localhost:9200")
index_name = "rag_index"

# Function to get embedding for query
def get_embedding(text):
    response = client.embeddings.create(
        input=[text],
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT")
    )
    return response.data[0].embedding

# Sample schema-level query
query = "What does the column CustomerSegment represent?"
embedding = get_embedding(query)

# Perform vector search
response = es.search(index=index_name, knn={
    "field": "embedding",
    "query_vector": embedding,
    "k": 5,
    "num_candidates": 20
})

# Display results
for hit in response["hits"]["hits"]:
    print(f"Score: {hit['_score']}")
    print(f"Type: {hit['_source'].get('type')}")
    print(f"Table: {hit['_source'].get('table')}")
    print(f"Name: {hit['_source'].get('name')}")
    print(f"Description: {hit['_source'].get('description')}")
    print("-" * 50)
