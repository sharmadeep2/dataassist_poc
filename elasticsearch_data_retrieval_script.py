from elasticsearch import Elasticsearch
es = Elasticsearch("http://localhost:9200")
res = es.search(index="rag_index", body={
    "query": {
        "match": {"'Period'[Fiscal Quarter]": "Q3"}
    }
})
print(res["hits"]["hits"])