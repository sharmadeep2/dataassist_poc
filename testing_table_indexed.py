from elasticsearch import Elasticsearch
import os

es = Elasticsearch(os.getenv("ES_ENDPOINT", "http://localhost:9200"))
ES_INDEX = "rag_index"

resp = es.search(index=ES_INDEX, size=1, query={"match_all": {}})
print(resp["hits"]["hits"])