import os
from elasticsearch import Elasticsearch

# Setup Elasticsearch connection
es = Elasticsearch(os.getenv("ES_ENDPOINT", "http://localhost:9200"))
ES_INDEX = "rag_index"

def get_all_table_names(index):
    # Aggregation to get unique table names
    resp = es.search(
        index=index,
        size=0,
        aggs={
            "tables": {
                "terms": {"field": "table_name.keyword", "size": 100}
            }
        }
    )
    return [bucket["key"] for bucket in resp["aggregations"]["tables"]["buckets"]]

def get_sample_doc_for_table(index, table_name):
    resp = es.search(
        index=index,
        size=1,
        query={"term": {"table_name.keyword": table_name}}
    )
    if resp["hits"]["hits"]:
        return resp["hits"]["hits"][0]["_source"]
    return None

if __name__ == "__main__":
    table_names = get_all_table_names(ES_INDEX)
    print(f"Found tables: {table_names}\n")
    for table in table_names:
        print(f"--- Sample document from table: {table} ---")
        doc = get_sample_doc_for_table(ES_INDEX, table)
        if doc:
            for k, v in doc.items():
                if k != "embedding":
                    print(f"{k}: {v}")
        else:
            print("No document found.")
        print("\n")