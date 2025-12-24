import os
import json
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

# Load schema file
with open("table_schema_updated_with_descriptions.json", "r", encoding="utf-8") as f:
    schema_data = json.load(f)

# Function to get embedding
def get_embedding(text):
    response = client.embeddings.create(
        input=[text],
        model=os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")
    )
    return response.data[0].embedding

# Ingest each table and its schema
for table in schema_data:
    table_name = table.get("table")
    table_meta = {
        "table": table.get("table"),
        "cube": table.get("cube"),
        "server": table.get("server"),
        "usecase": table.get("usecase"),
        "table_description": table.get("table_description")
    }

    # Embed table-level description
    table_embedding = get_embedding(table_meta["table_description"])

    # Index table-level metadata
    es.index(index=index_name, body={
        "type": "table",
        "metadata": table_meta,
        "embedding": table_embedding,
        "table_name": table_name
    })



    # Ingest columns and measures
    for field in table.get("schema", []):
        if "column_name" in field:
            name = field["column_name"]
        elif "measure" in field:
            name = field["measure"]
        else:
            continue

        description = field.get("description", "")
        embedding = get_embedding(description)

        es.index(index=index_name, body={
            "type": "field",
            "table": table.get("table"),
            "name": name,
            "description": description,
            "embedding": embedding
        })

print("✅ Data Assist schema ingested successfully.")
