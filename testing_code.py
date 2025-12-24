from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")  # Update if needed

# Initialize the scroll
page = es.search(
    #index="user_feedback",
    index="ragas_eval",
    body={"query": {"match_all": {}}},
    scroll='2m',  # Scroll context valid for 2 minutes
    size=100      # Fetch 100 records per batch (adjust as needed)
)

sid = page['_scroll_id']
scroll_size = len(page['hits']['hits'])

print(f"Total records: {es.count(index='user_feedback')['count']}")

# Process the first batch
for hit in page['hits']['hits']:
    print(hit['_source'])

# Keep scrolling until no more data
while scroll_size > 0:
    page = es.scroll(scroll_id=sid, scroll='2m')
    sid = page['_scroll_id']
    scroll_size = len(page['hits']['hits'])
    for hit in page['hits']['hits']:
        print(hit['_source'])
