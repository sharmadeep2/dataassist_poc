import os
import streamlit as st
import pandas as pd
import json
import openai
import re
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from openai import AzureOpenAI, BadRequestError
from typing import TypedDict, Optional, List, Dict, Any
from io import StringIO
from datetime import datetime


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
EXCEL_FILE = "dummy_data_output.xlsx"


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

# --- User Feedback Section (Elasticsearch Logging) ---
def log_user_feedback_es(es_client, index, query, confidence, feedback):
    doc = {
        "query": query,
        "confidence": confidence,
        "feedback": feedback,
        "timestamp": datetime.utcnow().isoformat()
    }
    try:
        es_client.index(index=index, document=doc)
    except Exception as e:
        st.error(f"Failed to log feedback to Elasticsearch: {e}")

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
    show_workflow("Planner")
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
    show_workflow("Retriever")
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
    show_workflow("Generator")
    query = state.get("query", "")
    context = state.get("context", "")
    prompt = (
        "Use the provided context to answer the user's query. "
        "If the context is insufficient or contains no relevant data or says “No ES results matched your filters,” the confidence score should be low (e.g., 0-30%)."
        "After your answer, provide a confidence score (0-100%) for your answer, based on the relevance and completeness of the context.  Format: \"Confidence: XX%\" "
        "If the provided data is insufficient, missing, or not relevant to the user's question, your confidence score should be low (e.g., 0-30%). Only give a high confidence score (e.g., 80-100%) if the answer is fully supported by the data."
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

st.set_page_config(page_title="Data Assist Q&A Assistant", layout="wide")
st.title("Data Assist Q&A Assistant")

tab1, tab2 = st.tabs(["Data Assist Q&A Assistant)", "Table Explorer"])

# --- Tab 1: RAG + Excel + LLM ---
with tab1:
    st.header("Ask a Question and Get Answers from Project Details Data")
    default_query = "List Project Details for Q3"
    user_query = st.text_input("Enter your question:", value="Show me the project details for Q3 FY24")

    def show_workflow(current_agent):
        # Define agent display order and icons
        agents = [
            ("User Query", "💬"),
            ("Planner", "📝"),
            ("Retriever", "🔍"),
            ("Generator", "🤖"),
            ("Answer", "✅"),
        ]
        # Colors
        active_color = "#4CAF50"  # Green
        inactive_color = "#e0e0e0"  # Light gray
        text_color = "#333"
        icon_size = "16px"
        font_size = "14px"
        bar_height = "4px"

        html = "<div style='display: flex; align-items: center; justify-content: center; margin-bottom: 8px;'>"
        for i, (name, icon) in enumerate(agents):
            is_active = (name == current_agent)
            html += f"""
            <div style='text-align: center; margin: 0 8px;'>
                <div style='
                    width: 40px; height: 40px; 
                    border-radius: 50%; 
                    background: {"#e8f5e9" if is_active else inactive_color};
                    display: flex; align-items: center; justify-content: center;
                    margin: 0 auto 2px auto;
                    border: 2px solid {"#388e3c" if is_active else inactive_color};
                    box-shadow: {"0 0 6px #b6f0c2" if is_active else "none"};
                '>
                    <span style='font-size: {icon_size};'>{icon}</span>
                </div>
                <span style='font-size: {font_size}; color: {active_color if is_active else text_color}; font-weight: {"bold" if is_active else "normal"};'>{name}</span>
            </div>
            """
            if i < len(agents) - 1:
                html += f"<div style='width: 32px; height: {bar_height}; background: {active_color if agents[i+1][0]==current_agent or is_active else inactive_color}; margin: 0 2px; border-radius: 2px;'></div>"
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)
#done
    def get_top_tables(user_query, top_n=3, min_confidence=2.0):
        """
        Returns a list of (table_name, confidence) tuples for tables relevant to the user_query.
        Uses ES _score as the confidence score.
        Only tables with _score >= min_confidence are returned.
        """
        try:
            resp = es.search(
                index=ES_INDEX,
                size=top_n,
                query={"match": {"metadata.table_description": user_query}},
                _source=["metadata.table", "metadata.confidence"]
            )
            tables = []
            for hit in resp["hits"]["hits"]:
                table_name = hit["_source"]["metadata"]["table"]
                score = hit.get("_score", 0)
                #st.write(f"Table: {table_name}, Score: {score}")
                if score >= min_confidence:
                    tables.append((table_name, score))
            return tables
        except Exception as e:
            st.error(f"Elasticsearch error: {e}")
            return []

    def extract_filters(user_query):
        fy_match = re.search(r"FY(\d{2})", user_query, re.IGNORECASE)
        q_match = re.search(r"Q([1-4])", user_query, re.IGNORECASE)
        fy = f"FY{fy_match.group(1)}" if fy_match else None
        q = f"Q{q_match.group(1)}" if q_match else None
        fq = f"{fy} - {q}" if fy and q else None
        return {"FY": fy, "Q": q, "FQ": fq}

    def filter_excel_data(table, filters):
        """
        Loads the best matching Excel sheet for the table and applies robust fiscal year and quarter filtering.
        Returns the filtered DataFrame and an error message (if any).
        """
        xl = pd.ExcelFile(EXCEL_FILE)
        sheet_names = xl.sheet_names
        norm_table = table.replace(" ", "").replace("_", "").lower()
        best_sheet = None

        # Try exact match first
        for s in sheet_names:
            norm_sheet = s.replace(" ", "").replace("_", "").lower()
            if norm_table == norm_sheet:
                best_sheet = s
                break

        # Try partial match if no exact match
        if not best_sheet:
            for s in sheet_names:
                norm_sheet = s.replace(" ", "").replace("_", "").lower()
                if norm_table in norm_sheet or norm_sheet in norm_table:
                    best_sheet = s
                    break

        if not best_sheet:
            st.warning(f"No matching sheet found for '{table}'. Available sheets: {sheet_names}")
            return None, f"No matching sheet found for '{table}'."

        df = xl.parse(best_sheet)

        # Identify fiscal year and quarter columns
        fy_col = None
        q_col = None
        for col in df.columns:
            if "year" in col.lower() or "fiscal" in col.lower():
                fy_col = col
            if "quarter" in col.lower() or "qtr" in col.lower():
                q_col = col

        # Apply fiscal year filter
        if fy_col and filters.get("FY"):
            fy_query = filters["FY"].replace("FY", "").replace(" ", "")
            # Try to match FY24 to 2024, FY 2024, 24, etc.
            fy_regex = re.compile(rf"{fy_query}$|{fy_query}", re.IGNORECASE)
            df = df[
                df[fy_col].astype(str).str.replace("FY", "", case=False).str.replace(" ", "").str.contains(fy_query, case=False, na=False) |
                df[fy_col].astype(str).str.contains(filters["FY"], case=False, na=False) |
                df[fy_col].astype(str).str.contains(fy_regex, na=False)
            ]

        # Apply quarter filter
        if q_col and filters.get("Q"):
            q_query = filters["Q"].replace("Q", "").replace(" ", "")
            # Try to match Q3 to 3, Qtr3, Quarter 3, etc.
            q_regex = re.compile(rf"{q_query}$|{q_query}", re.IGNORECASE)
            df = df[
                df[q_col].astype(str).str.replace("Q", "", case=False).str.replace(" ", "").str.contains(q_query, case=False, na=False) |
                df[q_col].astype(str).str.contains(filters["Q"], case=False, na=False) |
                df[q_col].astype(str).str.contains(q_regex, na=False)
            ]

        return df, None
#done

    def build_llm_prompt(user_query, table, df, table_confidence=None, confidence_threshold=2.0):
        """
        Only include table data in the LLM prompt if the table selection confidence is above the threshold
        and df is not empty.
        """
        if table_confidence is not None and table_confidence < confidence_threshold:
            context = "No relevant data was found for your query.\n\n"
        elif df is not None and not df.empty:
            preview_df = df.head(10)
            table_str = preview_df.to_markdown(index=False)
            context = f"Here is the relevant data from the '{table}' table:\n\n{table_str}\n\n"
        else:
            context = "No relevant data was found for your query.\n\n"

        prompt = (
            f"User asked: {user_query}\n"
            f"{context}"
            "Based on this data, answer the user's question in a concise, business-friendly way. "
            "If the data is sufficient, present your answer as a Markdown table. "
            "If the data is insufficient, say so clearly.\n"
            "After your answer, provide a confidence score (0-100%) for your answer, based on the relevance and completeness of the context. Format: \"Confidence: XX%\" and also include a brief explanation (4-5 sentences) of why you gave this confidence score, referencing the data or lack thereof.\n"
            "If the provided data is insufficient, missing, or not relevant to the user's question, your confidence score should be low (e.g., 0-30%). Only give a high confidence score (e.g., 80-100%) if the answer is fully supported by the data."

        )
        return prompt

    def extract_markdown_table(text):
        """
        Extracts the first Markdown table from the text.
        Returns the table as a string, or None if not found.
        Handles both standard and simple Markdown tables.
        """
        # Pattern for standard Markdown table: header, separator, at least one data row
        pattern = (
            r"((?:\|[^\n]+\|\n)"         # header row
            r"(?:\|[ \t:?\-]+?\|\n)"     # separator row (e.g., |---|---|)
            r"(?:\|[^\n]+\|\n)+)"        # at least one data row
        )
        match = re.search(pattern, text)
        if match:
            return match.group(1)

        # Fallback: any block of lines that look like a table (at least 2 rows)
        pattern_simple = r"((?:\|[^\n]+\|\n){2,})"
        match = re.search(pattern_simple, text)
        if match:
            return match.group(1)

        return None

    def markdown_table_to_df(md_table):
        """
        Converts a markdown table string to a pandas DataFrame.
        Removes the separator row (---) automatically.
        """
        # Split lines and remove the separator row (the second line)
        lines = md_table.strip().split('\n')
        # Check if the second line is a Markdown separator (---, :---, ---:, etc.)
        if len(lines) >= 2 and all(
            set(cell.strip()) <= set('-:| ') for cell in lines[1].split('|') if cell.strip()
        ):
            # Remove the separator row
            lines.pop(1)
        clean_table = '\n'.join(lines)
        # Now read as CSV
        df = pd.read_csv(StringIO(clean_table), sep="|", engine="python", skipinitialspace=True)
        # Remove empty columns (from leading/trailing pipes)
        df = df.loc[:, ~df.columns.str.match(r'Unnamed')]
        return df

    def show_llm_answer_table(llm_answer):
        md_table = extract_markdown_table(llm_answer)
        if md_table:
            df = markdown_table_to_df(md_table)
            # Custom header styling using st.dataframe's new features
            st.markdown(                        
                """
                <style>
                .stDataFrame thead tr th {
                    background-color: #e3f2fd !important;  /* Light blue */
                    color: #1565c0 !important;             /* Dark blue text for contrast */
                    font-weight: bold !important;
                    font-size: 15px !important;
                    border-bottom: 2px solid #90caf9 !important;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

            st.dataframe(df, width='stretch')
        else:
            st.write(llm_answer)  # fallback to plain text if no table found

    def ask_llm(prompt):
        try:
            # Create Azure OpenAI client
            client = openai.AzureOpenAI(
                api_key=AZURE_API_KEY,  # Your Azure OpenAI API key
                api_version=AZURE_API_VERSION,               # Use your deployed API version
                azure_endpoint=AZURE_ENDPOINT  # e.g. "https://YOUR_RESOURCE_NAME.openai.azure.com/"
            )
            response = client.chat.completions.create(
                model=CHAT_DEPLOYMENT,  # Your deployment name
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers using the provided table data."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"LLM error: {e}"

    def extract_confidence_and_explanation(llm_answer):
        # Extract confidence
        conf_match = re.search(r"Confidence[:\-]?\s*(\d{1,3})\s*%", llm_answer, re.IGNORECASE)
        confidence = int(conf_match.group(1)) if conf_match else None

        # Extract explanation (after "Explanation:" or similar)
        expl_match = re.search(r"Explanation[:\-]?\s*(.*)", llm_answer, re.IGNORECASE)
        explanation = expl_match.group(1).strip() if expl_match else "No explanation provided by the LLM."
        return confidence, explanation

    if st.button("Submit", key="rag_llm_submit"):
        #with st.spinner("Processing..."):
        try:
            result = lg_app.invoke({"query": user_query})
        except Exception as e:
            st.error(f"An error occurred while processing your request: {e}")
            st.stop()
        # Step 1: Query Elasticsearch for top tables
        def prioritize_tables_by_keywords(top_tables, user_query):
            keywords = [kw.lower() for kw in re.findall(r'\w+', user_query)]
            def score_table(table_name):
                return sum(kw in table_name.lower() for kw in keywords)
            return sorted(top_tables, key=lambda t: score_table(t[0]), reverse=True)

        # Usage:
        top_tables = get_top_tables(user_query, top_n=5, min_confidence=2.0)
        top_tables = prioritize_tables_by_keywords(top_tables, user_query)
        # Step 2: Planner - extract filters
        filters = extract_filters(user_query)
        if not top_tables:
            # No relevant table found, show "No relevant data" prompt
            llm_prompt = (
                f"User asked: {user_query}\n"
                "No relevant data was found for your query.\n\n"
                "Based on this data, answer the user's question in a concise, business-friendly way. "
                "If the data is sufficient, present your answer as a Markdown table. "
                "If the data is insufficient, say so clearly.\n"
                "After your answer, provide a confidence score (0-100%) for your answer, based on the relevance and completeness of the context. Format: \"Confidence: XX%\"\n"
                "If the provided data is insufficient, missing, or not relevant to the user's question, your confidence score should be low (e.g., 0-30%). Only give a high confidence score (e.g., 80-100%) if the answer is fully supported by the data."
            )
            # Optionally, display this prompt or pass it to your LLM
            st.info("No relevant table found for your query.")
            # You may want to stop further processing here
            st.stop()
        with st.spinner("Processing..."):
                table_name, table_confidence = top_tables[0]
                df, err = filter_excel_data(table_name, filters)
                llm_prompt = build_llm_prompt(user_query, table_name, df, table_confidence, confidence_threshold=2.0)

                # Step 3: Generator - load and filter Excel data
                df, err = filter_excel_data(top_tables[0][0], filters)
                if err:
                    st.error(err)
                    st.stop()
                if df is None or df.empty:
                    st.warning("No matching data found in Excel for your query and filters.")
                    st.stop()
                if table_confidence is not None and table_confidence > 0:
                    st.info("Here is the output data preview based on your query and filters:")     
                elif table_confidence == 0:
                    st.info("No relevant data found for your query, so no data preview is shown.")
                else:
                    st.info("Data may not be relevant to your query, so preview is hidden.")

                # Step 4: Build LLM prompt and get answer
                table_name, table_confidence = top_tables[0]
                llm_prompt = build_llm_prompt(user_query, table_name, df, table_confidence)
                llm_answer = ask_llm(llm_prompt)
                
                # Store last query and confidence in session state for feedback
                st.session_state["last_query"] = user_query
                st.session_state["last_confidence"] = confidence
                st.session_state["answer_shown"] = True  # Optional: to control feedback visibility

                # Step 5: Show answer
                #st.subheader("LLM Answer")
                show_llm_answer_table(llm_answer)

                # Optionally, show the LLM prompt for transparency
                with st.expander("Show LLM Prompt"):
                    st.code(llm_prompt, language="markdown") 
                
                # Extract and llm_response from the result
                llm_response = result.get("response", "") 
                #confidence = extract_confidence(llm_answer)
                confidence, explanation = extract_confidence_and_explanation(llm_answer)
                # Optionally, show the LLM Confidence for transparency

                if confidence is not None:
                    if confidence >= HIGH_CONFIDENCE:
                        st.success(f"**LLM Confidence Score:** {confidence}% (High)")
                        st.success(f"**Why this score?** {explanation}")
                        st.write(llm_response)
                    elif confidence >= LOW_CONFIDENCE:
                        st.warning(f"**LLM Confidence Score:** {confidence}% (Medium)")
                        st.warning(f"**Why this score?** {explanation}")
                        st.write(llm_response)
                        st.info("Consider verifying this answer or providing more details for higher accuracy.")
                    elif confidence >= VERY_LOW_CONFIDENCE:
                        st.error(f"**LLM Confidence Score:** {confidence}% (Low)")
                        st.error(f"**Why this score?** {explanation}")
                        st.write(llm_response)
                        st.info("This answer may be unreliable. Please rephrase your query or check your data sources.")
                    else:
                        st.error(f"**LLM Confidence Score:** {confidence}% (Very Low)")
                        st.error(f"**Why this score?** {explanation}")
                        st.info("No answer shown due to very low confidence. Please provide more details or try a different query.")
                else:
                    st.write(llm_response)
      
            # ... generate and show answer ...
                st.session_state["last_query"] = user_query
                st.session_state["last_confidence"] = confidence

                # Feedback section (always visible after answer)
                if "last_query" in st.session_state and "last_confidence" in st.session_state:
                    st.markdown("#### Was this answer helpful?")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("👍 Good Experience"):
                            log_user_feedback_es(es, "user_feedback", st.session_state["last_query"], st.session_state["last_confidence"], "thumbs_up")
                            st.success("Thank you for your feedback!")
                    with col2:
                        if st.button("👎 Bad Retrieval/Experience"):
                            log_user_feedback_es(es, "user_feedback", st.session_state["last_query"], st.session_state["last_confidence"], "thumbs_down")
                            st.warning("Thank you for your feedback!")
                
# --- Tab 2: Table Explorer ---
with tab2:
    st.header("Table Explorer")
    try:
        xl = pd.ExcelFile(EXCEL_FILE)
        sheet_names = xl.sheet_names
        selected_sheet = st.selectbox("Select a table (sheet):", sheet_names)
        df = xl.parse(selected_sheet)
        st.write(f"Preview of '{selected_sheet}':")
        st.dataframe(df.head(20))
        # Optionally, add search/filter functionality
        search_col = st.selectbox("Filter by column:", df.columns)
        search_val = st.text_input("Filter value:")
        if search_val:
            filtered_df = df[df[search_col].astype(str).str.contains(search_val, case=False, na=False)]
            st.write(f"Filtered preview ({search_col} contains '{search_val}'):")
            st.dataframe(filtered_df.head(20))
    except Exception as e:
        st.error(f"Error loading Excel file: {e}")
        st.write("Available sheets:", xl.sheet_names)
# --- End of app.py ---
