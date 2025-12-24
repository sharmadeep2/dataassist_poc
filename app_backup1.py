import os
import re
import streamlit as st
import pandas as pd
from typing import TypedDict, Optional, List, Dict, Any
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from openai import AzureOpenAI, BadRequestError
from langgraph.graph import StateGraph, END

# --- Page & environment setup ---
st.set_page_config(page_title="Project Details Q&A & Table Explorer", layout="wide")
load_dotenv()

# --- Read env once ---
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_INDEX = os.getenv("ES_INDEX", "rag_index")
CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4.1")
EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-ada-002")

HIGH_CONFIDENCE = 80
LOW_CONFIDENCE = 50
VERY_LOW_CONFIDENCE = 30

# --- Define clients at the top level ---
try:
    client = AzureOpenAI(
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
    )
except Exception as e:
    st.error(f"Azure OpenAI client error: {e}")
    st.stop()

try:
    es = Elasticsearch(ES_URL, request_timeout=30)
except Exception as e:
    st.error(f"Elasticsearch connection error: {e}")
    st.stop()

# --- State schema ---
class AgentState(TypedDict, total=False):
    query: str
    plan: Optional[str]
    context: Optional[str]
    response: Optional[str]
    response_error: Optional[str]

# --- Helper: parse FY & Quarter from user query ---
FY_RE = re.compile(r"\bFY(?P<fy>\d{2})\b", re.IGNORECASE)
Q_RE = re.compile(r"\bQ(?P<q>[1-4])\b", re.IGNORECASE)
def parse_fy_q(query: str) -> Dict[str, Optional[str]]:
    fy_match = FY_RE.search(query or "")
    q_match = Q_RE.search(query or "")
    fy = f"FY{fy_match.group('fy')}" if fy_match else None
    q = f"Q{q_match.group('q')}" if q_match else None
    fq_full = f"{fy} - {q}" if fy and q else None
    return {"fy": fy, "q": q, "fq_full": fq_full}


def extract_confidence(response: str):
    # Looks for "Confidence: XX%" (case-insensitive)
    match = re.search(r"Confidence[:\-]?\s*(\d{1,3})\s*%", response, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


# --- Planner ---
def planner(state: AgentState) -> AgentState:
    query = state.get("query") or ""
    state["plan"] = f"1) Retrieve relevant context\n2) Generate answer for: {query}"
    return state

# --- Retriever ---
RELEVANT_FIELDS: List[str] = [
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
    "content",
]
def retriever(state: AgentState) -> AgentState:
    query = state.get("query") or ""
    parsed = parse_fy_q(query)
    must_clauses: List[Dict[str, Any]] = []
    if parsed["fq_full"]:
        must_clauses.append({"match_phrase": {"'Period'[Fiscal Quarter]": parsed["fq_full"]}})
    elif parsed["q"]:
        must_clauses.append({"match_phrase": {"'Period'[Fiscal Quarter]": parsed["q"]}})
    if "project details" in query.lower():
        must_clauses.append({"match": {"content": "Project"}})
    #es_query = {"query": {"bool": {"must": must_clauses}}} if must_clauses else {"query": {"match_all": {}}}
    es_query = {"query": {"match_all": {}}}
    body = {
        **es_query,
        "size": 10,
        "_source": RELEVANT_FIELDS
    }
    try:
        res = es.search(index=ES_INDEX, body=body)
        hits = res.get("hits", {}).get("hits", [])
        chunks = []
        scores = []
        for h in hits:
            src = h.get("_source", {}) or {}
            score = h.get("_score", 0)
            scores.append(score)
            context_parts = [f"{field}: {src[field]}" for field in RELEVANT_FIELDS if field in src]
            if context_parts:
                chunks.append("; ".join(context_parts))
        state["context"] = "\n\n".join(chunks) if chunks else "No ES results matched your filters."
        state["scores"] = scores
    except Exception as e:
        state["context"] = f"Retriever error: {e}"
    return state

# --- Generator via Chat Completions ---
def generator(state: AgentState) -> AgentState:
    query = state.get("query", "")
    context = state.get("context", "")
    prompt = (
        "Use the provided context to answer the user's query. "
        "If the context is insufficient or contains no relevant data or says “No ES results matched your filters,” the confidence score should be low (e.g., 0-30%)."
        "After your answer, provide a confidence score (0-100%) for your answer, based on the relevance and completeness of the context.  Format: \"Confidence: XX%\" "
    )
    try:
        resp = client.chat.completions.create(
            model=CHAT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are a concise, accurate assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        state["response"] = resp.choices[0].message.content
    except BadRequestError as e:
        msg = str(e)
        state["response_error"] = (
            "Generation failed (BadRequest). Your Azure resource may require the Responses API.\n"
            "Options to fix:\n"
            " 1) Upgrade 'openai' package: pip install -U openai>=1.44 (adds client.responses)\n"
            " 2) Switch AZURE_OPENAI_API_VERSION to one that supports Chat Completions (e.g., 2024-02-15)\n"
            f"Details: {msg}"
        )
    except Exception as e:
        state["response_error"] = f"Generation failed: {e}"
    return state

# --- Build & compile the graph ---
workflow = StateGraph(AgentState)
workflow.add_node("planner", planner)
workflow.add_node("retriever", retriever)
workflow.add_node("generator", generator)
workflow.set_entry_point("planner")
workflow.add_edge("planner", "retriever")
workflow.add_edge("retriever", "generator")
workflow.add_edge("generator", END)
lg_app = workflow.compile()

# --- Table Explorer Utilities ---
@st.cache_data(show_spinner=False)
def get_all_table_names(index):
    try:
        resp = es.search(
            index=index,
            size=0,
            aggs={"tables": {"terms": {"field": "table_name.keyword", "size": 100}}}
        )
        return [bucket["key"] for bucket in resp["aggregations"]["tables"]["buckets"]]
    except Exception:
        return []

@st.cache_data(show_spinner=False)
def get_doc_count_for_table(index, table_name):
    try:
        resp = es.count(
            index=index,
            body={"query": {"term": {"table_name.keyword": table_name}}}
        )
        return resp["count"]
    except Exception:
        return 0

@st.cache_data(show_spinner=False)
def get_docs_for_table(index, table_name, size=10, from_=0, filters=None):
    query = {"term": {"table_name.keyword": table_name}}
    if filters:
        must = [query]
        for k, v in filters.items():
            must.append({"term": {k: v}})
        query = {"bool": {"must": must}}
    try:
        resp = es.search(
            index=index,
            size=size,
            from_=from_,
            query=query
        )
        return [hit["_source"] for hit in resp["hits"]["hits"]]
    except Exception:
        return []

def get_all_fields(table_name):
    docs = get_docs_for_table(ES_INDEX, table_name, size=1)
    if docs:
        return list(docs[0].keys())
    return []

def export_to_csv(docs, fields):
    df = pd.DataFrame(docs)
    if "embedding" in df.columns:
        df = df.drop(columns=["embedding"])
    if fields:
        df = df[fields]
    return df.to_csv(index=False).encode("utf-8")

# --- Streamlit Tabs UI ---
tab1, tab2 = st.tabs(["Q&A Assistant", "Table Explorer"])

with tab1:
    st.header("Project Details Q&A Assistant")
    default_query = "List Project Details for Q3"
    user_query = st.text_input("Enter your query:", value=default_query, help="Try: 'List Project Details for FY24 - Q3'")
    if st.button("Submit", key="qa_submit"):
        with st.spinner("Running agents..."):
            try:
                result = lg_app.invoke({"query": user_query})
            except Exception as e:
                st.error(f"An error occurred while processing your request: {e}")
                st.stop()

            st.subheader("Agent Plan")
            st.code(result.get("plan", ""), language="text")

            st.subheader("Response")
            if result.get("response"):
                st.write(result["response"])
            else:
                st.warning("No response generated. Try broadening your query or check your data.")

            if result.get("response_error"):
                st.error(result["response_error"])

            with st.expander("Retrieved Context (for debugging)"):
                st.text(result.get("context", ""))

            # Display context as a table
            with st.expander("Project Details Table"):
                context = result.get("context", "")
                rows = []
                for chunk in context.split("\n\n"):
                    row = {}
                    for part in chunk.split("; "):
                        if ": " in part:
                            key, value = part.split(": ", 1)
                            row[key] = value
                    if row:
                        rows.append(row)
                if rows:
                    df = pd.DataFrame(rows)
                    st.dataframe(df, width='stretch', height=300)
                else:
                    st.info("No project details found for this query.")
            # Extract and display confidence score
            llm_response = result.get("response", "")
            confidence = extract_confidence(llm_response)
            
            # Adjust confidence if context indicates no results
            if ("no es results matched your filters" in context.lower() or
                "no data found" in llm_response.lower()):
                if confidence is not None:
                    confidence = min(confidence, 30)
                else:
                    confidence = 0  # or 30, or whatever you prefer as the default for "no data"

            #st.write(llm_response)
            if confidence is not None:
                if confidence >= HIGH_CONFIDENCE:
                    st.success(f"LLM Confidence Score: {confidence}% (High)")
                    st.write(llm_response)
                elif confidence >= LOW_CONFIDENCE:
                    st.warning(f"LLM Confidence Score: {confidence}% (Medium)")
                    st.write(llm_response)
                    st.info("Consider verifying this answer or providing more details for higher accuracy.")
                elif confidence >= VERY_LOW_CONFIDENCE:
                    st.error(f"LLM Confidence Score: {confidence}% (Low)")
                    st.write(llm_response)
                    st.info("This answer may be unreliable. Please rephrase your query or check your data sources.")
                else:
                    st.error(f"LLM Confidence Score: {confidence}% (Very Low)")
                    st.info("No answer shown due to very low confidence. Please provide more details or try a different query.")
            else:
                st.write(llm_response)
            
with tab2:
    st.header("Table Explorer")
    table_names = get_all_table_names(ES_INDEX)
    if not table_names:
        st.error("No tables found. Please check your index and ingestion.")
        st.stop()

    search_table = st.text_input("Search table name (partial or full):", key="table_search")
    filtered_tables = [t for t in table_names if search_table.lower() in t.lower()] if search_table else table_names
    table = st.selectbox("Select a table to explore:", filtered_tables)

    if table:
        total_docs = get_doc_count_for_table(ES_INDEX, table)
        st.write(f"**Total documents:** {total_docs}")

        # Pagination
        page_size = st.number_input("Rows per page", min_value=1, max_value=100, value=10, key="page_size")
        max_page = (total_docs - 1) // page_size + 1
        page = st.number_input("Page number", min_value=1, max_value=max_page, value=1, key="page_num")
        from_ = (page - 1) * page_size

        # Field filtering
        all_fields = get_all_fields(table)
        default_fields = [f for f in all_fields if f != "embedding"]
        fields = st.multiselect("Fields to display", options=default_fields, default=default_fields, key="fields")

        # Search/filter by field value
        filter_expander = st.expander("Add field filters (optional)")
        filters = {}
        with filter_expander:
            for field in default_fields:
                val = st.text_input(f"Filter by {field}", key=f"filter_{field}")
                if val:
                    filters[field] = val

        # Retrieve and display data
        docs = get_docs_for_table(ES_INDEX, table, size=page_size, from_=from_, filters=filters)
        if docs:
            df = pd.DataFrame(docs)
            if "embedding" in df.columns:
                df = df.drop(columns=["embedding"])
            if fields:
                df = df[fields]
            #st.dataframe(df, use_container_width=True)
            st.dataframe(df, width='stretch', height=300)
        else:
            st.warning("No data found for this page/filters.")

        # Export to CSV
        if docs and st.button("Export this page to CSV", key="export_csv"):
            csv = export_to_csv(docs, fields)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"{table}_page{page}.csv",
                mime="text/csv"
            )

        st.info("Tip: Use the search box to quickly find a table. Use the expander to filter by field values. Use the multiselect to customize columns. Use the page controls to navigate large tables. Export any page to CSV for further analysis.")