import os
import random
import json
from elasticsearch import Elasticsearch
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# Setup
es = Elasticsearch(os.getenv("ES_ENDPOINT", "http://localhost:9200"))
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)
EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")
ES_INDEX = "rag_index"

# Load schema
with open("table_schema_updated_with_descriptions.json", "r", encoding="utf-8") as f:
    schema = json.load(f)

def get_sample_value(col):
    # Use sample values if available
    if "sample_values" in col and col["sample_values"]:
        return random.choice([v for v in col["sample_values"] if v not in [None, "null", ""]])
    dtype = col.get("dtype", col.get("type", "")).lower()
    # Fallbacks based on type
    if "int" in dtype:
        return random.randint(1, 100)
    elif "double" in dtype or "float" in dtype or "real" in dtype:
        return round(random.uniform(1, 100), 2)
    elif "date" in dtype or "datetime" in dtype:
        return "2024-01-01 00:00:00"
    elif "string" in dtype or "char" in dtype or "text" in dtype:
        return f"Sample {col.get('column_name', col.get('name', 'Field'))}"
    else:
        return f"Sample {col.get('column_name', col.get('name', 'Field'))}"

for table in schema:
    table_name = table.get("table") or table.get("table_name")
    columns = table.get("schema", table.get("columns", []))
    doc = {}
    for col in columns:
        # Some entries are measures, skip if not a column
        col_name = col.get("column_name") or col.get("name")
        if not col_name:
            continue
        doc[col_name] = get_sample_value(col)
    doc["table_name"] = table_name
    # Add a content field for embedding
    content = f"Sample data for table {table_name}: " + ", ".join([f"{k}={v}" for k, v in doc.items()])
    embedding_response = client.embeddings.create(
        input=content,
        model=EMBED_DEPLOYMENT
    )
    embedding = embedding_response.data[0].embedding
    doc["content"] = content
    doc["embedding"] = embedding
    
    # Index the document
    es.index(index=ES_INDEX, document=doc)
    print(f"Indexed sample document for table: {table_name}")