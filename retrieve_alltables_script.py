import os
from elasticsearch import Elasticsearch

# Setup Elasticsearch connection (adjust if needed)
es = Elasticsearch(os.getenv("ES_ENDPOINT", "http://localhost:9200"))
ES_INDEX = "rag_index"

def print_sample_docs(index, size=10):
    print(f"Fetching {size} sample documents from index '{index}'...\n")
    resp = es.search(index=index, size=size, query={"match_all": {}})
    for i, hit in enumerate(resp["hits"]["hits"], 1):
        doc = hit["_source"]
        # Try to extract table name from content or keys
        table_name = "Unknown"
        if "content" in doc and "table" in doc["content"]:
            table_name = doc["content"].split("table ")[1].split(":")[0]
        print(f"--- Document {i} (Table: {table_name}) ---")
        for k, v in doc.items():
            if k != "embedding":  # Don't print embedding vector
                print(f"{k}: {v}")
        print("\n")

if __name__ == "__main__":
    print_sample_docs(ES_INDEX, size=10)