import os
import streamlit as st
import pandas as pd
from elasticsearch import Elasticsearch

# --- CONFIGURATION ---
ES_INDEX = "rag_index"
ES_ENDPOINT = os.getenv("ES_ENDPOINT", "http://localhost:9200")
es = Elasticsearch(ES_ENDPOINT)

# --- UTILITY FUNCTIONS ---
@st.cache_data(show_spinner=False)
def get_all_table_names(index):
    resp = es.search(
        index=index,
        size=0,
        aggs={"tables": {"terms": {"field": "table_name.keyword", "size": 100}}}
    )
    return [bucket["key"] for bucket in resp["aggregations"]["tables"]["buckets"]]

@st.cache_data(show_spinner=False)
def get_doc_count_for_table(index, table_name):
    resp = es.count(
        index=index,
        body={"query": {"term": {"table_name.keyword": table_name}}}
    )
    return resp["count"]

@st.cache_data(show_spinner=False)
def get_docs_for_table(index, table_name, size=10, from_=0, filters=None):
    query = {"term": {"table_name.keyword": table_name}}
    if filters:
        must = [query]
        for k, v in filters.items():
            must.append({"term": {k: v}})
        query = {"bool": {"must": must}}
    resp = es.search(
        index=index,
        size=size,
        from_=from_,
        query=query
    )
    return [hit["_source"] for hit in resp["hits"]["hits"]]

def get_all_fields(table_name):
    docs = get_docs_for_table(ES_INDEX, table_name, size=1)
    if docs:
        return list(docs[0].keys())
    return []

def filter_docs_by_field(docs, filters):
    if not filters:
        return docs
    filtered = []
    for doc in docs:
        match = True
        for k, v in filters.items():
            if str(doc.get(k, "")).lower() != str(v).lower():
                match = False
                break
        if match:
            filtered.append(doc)
    return filtered

def export_to_csv(docs, fields):
    df = pd.DataFrame(docs)
    if "embedding" in df.columns:
        df = df.drop(columns=["embedding"])
    if fields:
        df = df[fields]
    return df.to_csv(index=False).encode("utf-8")

# --- STREAMLIT APP ---
st.set_page_config(page_title="Elasticsearch Table Explorer", layout="wide")
st.title("🔎 Elasticsearch Table Explorer")

# 1. Table selection
table_names = get_all_table_names(ES_INDEX)
if not table_names:
    st.error("No tables found. Please check your index and ingestion.")
    st.stop()

search_table = st.text_input("Search table name (partial or full):")
filtered_tables = [t for t in table_names if search_table.lower() in t.lower()] if search_table else table_names
table = st.selectbox("Select a table to explore:", filtered_tables)

if table:
    total_docs = get_doc_count_for_table(ES_INDEX, table)
    st.write(f"**Total documents:** {total_docs}")

    # 2. Pagination
    page_size = st.number_input("Rows per page", min_value=1, max_value=100, value=10)
    max_page = (total_docs - 1) // page_size + 1
    page = st.number_input("Page number", min_value=1, max_value=max_page, value=1)
    from_ = (page - 1) * page_size

    # 3. Field filtering
    all_fields = get_all_fields(table)
    default_fields = [f for f in all_fields if f != "embedding"]
    fields = st.multiselect("Fields to display", options=default_fields, default=default_fields)

    # 4. Search/filter by field value
    filter_expander = st.expander("Add field filters (optional)")
    filters = {}
    with filter_expander:
        for field in default_fields:
            val = st.text_input(f"Filter by {field}", key=f"filter_{field}")
            if val:
                filters[field] = val

    # 5. Retrieve and display data
    docs = get_docs_for_table(ES_INDEX, table, size=page_size, from_=from_, filters=filters)
    if docs:
        df = pd.DataFrame(docs)
        if "embedding" in df.columns:
            df = df.drop(columns=["embedding"])
        if fields:
            df = df[fields]
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("No data found for this page/filters.")

    # 6. Export to CSV
    if docs and st.button("Export this page to CSV"):
        csv = export_to_csv(docs, fields)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"{table}_page{page}.csv",
            mime="text/csv"
        )

    # 7. Error handling and user guidance is built-in via Streamlit widgets and warnings

    # 8. Web UI is provided by Streamlit

st.info("Tip: Use the search box to quickly find a table. Use the expander to filter by field values. Use the multiselect to customize columns. Use the page controls to navigate large tables. Export any page to CSV for further analysis.")