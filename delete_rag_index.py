from elasticsearch import Elasticsearch
import os

es = Elasticsearch(os.getenv("ES_ENDPOINT", "http://localhost:9200"))
ES_INDEX = "rag_index"

# Delete the index (this removes all documents and the mapping)
if es.indices.exists(index=ES_INDEX):
    es.indices.delete(index=ES_INDEX)
    print(f"Index '{ES_INDEX}' deleted.")
else:
    print(f"Index '{ES_INDEX}' does not exist.")