import os
from elasticsearch import Elasticsearch
import pandas as pd

es = Elasticsearch(os.getenv("ES_ENDPOINT", "http://localhost:9200"))
ES_INDEX = "rag_index"

def get_all_table_names(index):
    resp = es.search(
        index=index,
        size=0,
        aggs={"tables": {"terms": {"field": "table_name.keyword", "size": 100}}}
    )
    return [bucket["key"] for bucket in resp["aggregations"]["tables"]["buckets"]]

def get_docs_for_table(index, table_name, size=10):
    resp = es.search(
        index=index,
        size=size,
        query={"term": {"table_name.keyword": table_name}}
    )
    return [hit["_source"] for hit in resp["hits"]["hits"]]

if __name__ == "__main__":
    table_names = get_all_table_names(ES_INDEX)
    print("Available tables:", table_names)
    table = input("Enter table name to view (or leave blank to list all): ").strip()
    if table and table in table_names:
        docs = get_docs_for_table(ES_INDEX, table, size=10)
        if docs:
            df = pd.DataFrame(docs)
            print(df.drop(columns=["embedding"], errors="ignore").head())
        else:
            print("No data found for this table.")
    else:
        print("Listing one sample from each table:")
        for t in table_names:
            docs = get_docs_for_table(ES_INDEX, t, size=1)
            if docs:
                print(f"\nTable: {t}")
                df = pd.DataFrame(docs)
                print(df.drop(columns=["embedding"], errors="ignore").head())
