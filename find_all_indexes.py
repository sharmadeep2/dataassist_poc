from elasticsearch import Elasticsearch

# Connect to your Elasticsearch server
es = Elasticsearch("http://localhost:9200")

# List all index names (correct usage: keyword argument)
indices = es.indices.get_alias(index="*")
print("Elasticsearch indices:")
for index_name in indices.keys():
    print(index_name)