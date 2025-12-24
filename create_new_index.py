from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")  # Update if needed

# Create the index with a basic mapping (optional)
mapping = {
    "mappings": {
        "properties": {
            "query": {"type": "text"},
            "confidence": {"type": "integer"},
            "feedback": {"type": "keyword"},
            "timestamp": {"type": "date"}
        }
    }
}

es.indices.create(index="user_feedback", body=mapping, ignore=400)
print("Index created (or already exists).")