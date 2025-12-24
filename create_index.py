from elasticsearch import Elasticsearch

# Connect to local Elasticsearch
es = Elasticsearch("http://localhost:9200")

# Define index name
index_name = "rag_index"

# Define mapping for RAG
mapping = {
    "mappings": {
        "properties": {
            "content": {"type": "text"},
            "embedding": {"type": "dense_vector", "dims": 1536}
        }
    }
}

# Create index if it doesn't exist
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name, body=mapping)
    print(f"Index '{index_name}' created successfully.")
else:
    print(f"Index '{index_name}' already exists.")