
from elasticsearch import Elasticsearch
es = Elasticsearch("http://localhost:9200")
resp = es.search(index="rag_index", size=5)
for hit in resp["hits"]["hits"]:
    print(hit["_source"])
'''
#Count the number of documents in the index
from elasticsearch import Elasticsearch
es = Elasticsearch("http://localhost:9200")
print(es.count(index="rag_index"))
'''
'''
from elasticsearch import Elasticsearch
es = Elasticsearch("http://localhost:9200")
resp = es.search(index="rag_index", size=5)
for hit in resp["hits"]["hits"]:
    print(hit["_source"])
'''
