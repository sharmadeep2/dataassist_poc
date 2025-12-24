import os
from elasticsearch import Elasticsearch
import pandas as pd

ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_INDEX = os.getenv("ES_INDEX", "rag_index")
es = Elasticsearch(ES_URL)

# Set your ES endpoint and index
def fetch_project_details_docs(index, table_name="Project Details", size=20):
    resp = es.search(
        index=index,
        size=size,
        query={"term": {"table_name.keyword": table_name}}
    )
    docs = [hit["_source"] for hit in resp["hits"]["hits"]]
    return docs

if __name__ == "__main__":
    docs = fetch_project_details_docs(ES_INDEX)
    if not docs:
        print("No data found for 'Project Details' table.")
    else:
        df = pd.DataFrame(docs)
        print(f"Found {len(df)} documents in 'Project Details' table.\n")
        print(df.head(10).to_string(index=False))


