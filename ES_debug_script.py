from elasticsearch import Elasticsearch

ES_INDEX = "rag_index"  # or your actual index name

body = {
    "query": {
        "match": {"'Period'[Fiscal Quarter]": "Q3"}
    },
    "size": 5,
    "_source": [
        "Project Details[Master Customer Code]",
        "Project Details[Service Line Unit]",
        "Project Details[Service Line Unit 2]",
        "Project Details[Project PU]",
        "Master Project Code",
        "Project Details[Revenue Credit Unit]",
        "Project Details[Contract Type]",
        "'Period'[Month]",
        "'Period'[txtFiscalYear]",
        "'Period'[Fiscal Quarter]",
        "sales_amount",
        "content"
    ]
}

es = Elasticsearch("http://localhost:9200")
res = es.search(index=ES_INDEX, body=body)
print(res["hits"]["hits"])