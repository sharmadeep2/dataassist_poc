from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")
doc = {
    "query": "test query",
    "confidence": 99,
    "feedback": "thumbs_up",
    "timestamp": "2025-10-01T12:00:00"
}
res = es.index(index="user_feedback", document=doc)
print("Inserted:", res)