import os
import re
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from openai import AzureOpenAI, BadRequestError

# --- Page & environment setup ---
st.set_page_config(page_title="Project Details Q&A & Table Explorer", layout="wide")
load_dotenv()

# --- Read env once ---
AZURE_API_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_INDEX = os.getenv("ES_INDEX", "rag_index")
CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4.1")

HIGH_CONFIDENCE = 80
LOW_CONFIDENCE = 50
VERY_LOW_CONFIDENCE = 30

def extract_confidence(response: str):
    # Looks for "Confidence: XX%" (case-insensitive)
    match = re.search(r"Confidence[:\-]?\s*(\d{1,3})\s*%", response, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None
# --- Helper: parse FY & Quarter from user query ---
FY_RE = re.compile(r"\bFY(?P<fy>\d{2})\b", re.IGNORECASE)
Q_RE = re.compile(r"\bQ(?P<q>[1-4])\b", re.IGNORECASE)

def parse_fy_q(query: str):
    fy_match = FY_RE.search(query or "")
    q_match = Q_RE.search(query or "")
    fy = f"FY{fy_match.group('fy')}" if fy_match else None
    q = f"Q{q_match.group('q')}" if q_match else None
    fq_full = f"{fy} - {q}" if fy and q else None
    return {"fy": fy, "q": q, "fq_full": fq_full}

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

# --- Q&A Assistant ---
tab1, tab2 = st.tabs(["Q&A Assistant", "Table Explorer"])
with tab1:
    st.header("Project Details Q&A Assistant")
    default_query = "List Project Details for Q3"
    user_query = st.text_input("Enter your query:", value=default_query, help="Try: 'List Project Details for FY24 - Q3'")
    if st.button("Submit", key="qa_submit"):
        with st.spinner("Running agents..."):
            # --- Planner ---
            plan = f"1) Retrieve relevant context\n2) Generate answer for: {user_query}"

            # --- Retriever ---
            parsed = parse_fy_q(user_query)
            must_clauses = []
            if parsed["q"]:
                must_clauses.append({"match": {"FiscalQuarter": parsed["q"]}})
            if parsed["fy"]:
                must_clauses.append({"match": {"FiscalYear": parsed["fy"]}})
            if "project details" in user_query.lower():
                must_clauses.append({"match": {"content": "Project"}})

            es_query = {"query": {"bool": {"must": must_clauses}}} if must_clauses else {"query": {"match_all": {}}}
            body = {
                **es_query,
                "size": 10,
                "_source": [
                    "MasterCustomerCode",
                    "ServiceLineUnit",
                    "ProjectPU",
                    "MasterProjectCode",
                    "RevenueCreditUnit",
                    "ContractType",
                    "Month",
                    "FiscalYear",
                    "FiscalQuarter",
                    "sales_amount",
                    "content",
                ]
            }

            try:
                st.write("Parsed filters:", parsed)
                st.write("Must clauses:", must_clauses)
                st.write("ES Query Body:", body)
                res = es.search(index=ES_INDEX, body=body)
                st.write("ES Search Results:", res)
                hits = res.get("hits", {}).get("hits", [])
                chunks = []
                scores = []
                for h in hits:
                    src = h.get("_source", {}) or {}
                    score = h.get("_score", 0)
                    scores.append(score)
                    context_parts = [f"{field}: {src[field]}" for field in body["_source"] if field in src]
                    if context_parts:
                        chunks.append("; ".join(context_parts))
                context = "\n\n".join(chunks) if chunks else "No ES results matched your filters."
            except Exception as e:
                context = f"Retriever error: {e}"
            
            test_res = es.search(index=ES_INDEX, body={"query": {"match_all": {}}, "size": 1})
            st.write("Sample ES Document:", test_res["hits"]["hits"][0]["_source"])
            
            # --- Generator via Chat Completions ---
            prompt = (
                "Use the provided context to answer the user's query. "
                "If the context is insufficient or contains no relevant data or says “No ES results matched your filters,” the confidence score should be low (e.g., 0-30%)."
                "After your answer, provide a confidence score (0-100%) for your answer, based on the relevance and completeness of the context. Format: \"Confidence: XX%\" "
            )
            try:
                resp = client.chat.completions.create(
                    model=CHAT_DEPLOYMENT,
                    messages=[
                        {"role": "system", "content": "You are a concise, accurate assistant."},
                        {"role": "user", "content": prompt + "\n\nContext:\n" + context + "\n\nUser Query:\n" + user_query},
                    ],
                    temperature=0.2,
                )
                llm_response = resp.choices[0].message.content
            except BadRequestError as e:
                llm_response = (
                    "Generation failed (BadRequest). Your Azure resource may require the Responses API.\n"
                    "Options to fix:\n"
                    " 1) Upgrade 'openai' package: pip install -U openai>=1.44 (adds client.responses)\n"
                    " 2) Switch AZURE_OPENAI_API_VERSION to one that supports Chat Completions (e.g., 2024-02-15)\n"
                    f"Details: {str(e)}"
                )
            except Exception as e:
                llm_response = f"Generation failed: {e}"

            # --- Display Results ---
            st.subheader("Agent Plan")
            st.code(plan, language="text")
            st.subheader("Response")
            if llm_response:
                st.write(llm_response)
            else:
                st.warning("No response generated. Try broadening your query or check your data.")

            with st.expander("Retrieved Context (for debugging)"):
                st.text(context)

            # Display context as a table
            with st.expander("Project Details Table"):
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
            confidence = extract_confidence(llm_response)
            # Adjust confidence if context indicates no results
            if ("no es results matched your filters" in context.lower() or
                "no data found" in llm_response.lower()):
                if confidence is not None:
                    confidence = min(confidence, 30)
                else:
                    confidence = 0  # or 30, or whatever you prefer as the default for "no data"
            if confidence is not None:
                if confidence >= HIGH_CONFIDENCE:
                    st.success(f"LLM Confidence Score: {confidence}% (High)")
                elif confidence >= LOW_CONFIDENCE:
                    st.warning(f"LLM Confidence Score: {confidence}% (Medium)")
                    st.info("Consider verifying this answer or providing more details for higher accuracy.")
                elif confidence >= VERY_LOW_CONFIDENCE:
                    st.error(f"LLM Confidence Score: {confidence}% (Low)")
                    st.info("This answer may be unreliable. Please rephrase your query or check your data sources.")
                else:
                    st.error(f"LLM Confidence Score: {confidence}% (Very Low)")
                    st.info("No answer shown due to very low confidence. Please provide more details or try a different query.")

with tab2:
    st.header("Table Explorer")
    # Table Explorer logic from your previous version
    # ... (keep your existing Table Explorer code here, unchanged)
    # For brevity, not repeated here, but you can copy from your working app.py or app-bckup.pyHIGH_CONFIDENCE = 80
LOW_CONFIDENCE = 50
VERY_LOW_CONFIDENCE = 30





