import os
import requests
import re
import json
from xml.parsers.expat import model
import streamlit as st
import pandas as pd
import csv
import uuid
import logging
# from aiopslab import Orchestrator, GPT4  # Removed because aiopslab is not available
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from elasticsearch import Elasticsearch
import openai
from datetime import datetime, timezone
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from openai import AzureOpenAI, BadRequestError
from typing import TypedDict, Optional, List, Dict, Any
from io import StringIO
from datetime import datetime
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from datasets import Dataset

# --- Page & environment setup ---
st.set_page_config(page_title="Project Details Q&A & Table Explorer", layout="wide")
load_dotenv()

# Code for Logging and AIOpsLab integratoin start here
# =============================
# Modular, Robust Event Logging
# ==============================
# --- Configure Python logger (to console & file) ---

LOG_DIR = Path("./logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "app_runtime.log", encoding="utf-8"),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger("poc-logging")

# --- Config for sinks (toggle via env if needed) ---
ENABLE_FILE_SINK = True
ENABLE_ES_SINK = True            # requires a live es client
ENABLE_API_SINK = False          # set True if you implement send_to_api()

JSONL_PATH = LOG_DIR / "events.jsonl"
CSV_PATH = LOG_DIR / "events.csv"

# --- Optional: external AIOPSLab ingestion endpoint (if used)
# AIOPSLAB_INGEST_URL = os.getenv("AIOPSLAB_INGEST_URL")   # e.g., https://aiops/acme/ingest
# AIOPSLAB_TOKEN = os.getenv("AIOPSLAB_TOKEN")             # e.g., bearer token for API


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_primitive(value: Any) -> Any:
    """
    Coerce complex objects to JSON-safe primitives for CSV/JSONL.
    """
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_safe_primitive(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _safe_primitive(v) for k, v in value.items()}
    # Fallback to string for anything else (e.g., numpy, pd types)
    return str(value)


def _coerce_event(
    event_type: str,
    query: Optional[str] = None,
    table_confidence: Optional[Union[int, float]] = None,
    llm_score: Optional[Union[int, float]] = None,
    explanation: Optional[str] = None,
    user_feedback: Optional[str] = None,
    composite_reward: Optional[float] = None,
    faithfulness: Optional[float] = None,
    answer_relevancy: Optional[float] = None,
    context_count: Optional[int] = None,
    doc_ids: Optional[List[str]] = None,
    extra: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a normalized, schema-safe event.
    """
    payload: Dict[str, Any] = {
        "event_id": str(uuid.uuid4()),
        "event_type": event_type,
        "timestamp": _now_iso(),
        "query": query or "",
        "table_confidence": float(table_confidence) if table_confidence is not None else None,
        "llm_score": float(llm_score) if llm_score is not None else None,
        "explanation": explanation or "",
        "user_feedback": user_feedback or "",
        "composite_reward": float(composite_reward) if composite_reward is not None else None,
        "faithfulness": float(faithfulness) if faithfulness is not None else None,
        "answer_relevancy": float(answer_relevancy) if answer_relevancy is not None else None,
        "context_count": int(context_count) if context_count is not None else None,
        "doc_ids": doc_ids or [],
        "error": error or "",
        "app_version": "poc-rlhf-lite-v1",
    }
    if extra:
        payload["extra"] = _safe_primitive(extra)
    # Ensure JSON-safe coercion
    return {k: _safe_primitive(v) for k, v in payload.items()}


def _write_jsonl(path: Path, event: Dict[str, Any]) -> None:
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.exception("JSONL write failed: %s", e)


def _write_csv(path: Path, event: Dict[str, Any]) -> None:
    try:
        is_new = not path.exists()
        with path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(event.keys()))
            if is_new:
                writer.writeheader()
            writer.writerow(event)
    except Exception as e:
        logger.exception("CSV write failed: %s", e)


def _index_to_es(es_client: Optional[Elasticsearch], index: str, event: Dict[str, Any]) -> None:
    if not (ENABLE_ES_SINK and es_client):
        return
    try:
        es_client.options(request_timeout=3).index(index=index, document=event)
    except Exception as e:
        # Do not break UX; just log the error
        logger.warning("ES index failed: %s", e)


def _send_to_api(event: Dict[str, Any]) -> None:
    """
    Optional: send event to external AIOPSLab or analytics endpoint.
    Requires 'requests'. Toggle ENABLE_API_SINK True and set env vars.
    """
    if not ENABLE_API_SINK or not AIOPSLAB_INGEST_URL:
        return
    try:
        import requests  # local import to avoid module requirement if unused
        headers = {
            "Authorization": f"Bearer {AIOPSLAB_TOKEN}" if AIOPSLAB_TOKEN else "",
            "Content-Type": "application/json",
        }
        resp = requests.post(AIOPSLAB_INGEST_URL, headers=headers, json=event, timeout=3)
        if resp.status_code >= 300:
            logger.warning("API sink rejected event: %s | %s", resp.status_code, resp.text)
    except Exception as e:
        logger.warning("API sink failed: %s", e)


def log_event(
    event_type: str,
    *,
    es_client: Optional[Elasticsearch] = None,
    es_index: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Build and persist a normalized event to multiple sinks.
    Returns the event dict (for in-memory debugging/tests).
    """
    event = _coerce_event(event_type=event_type, **kwargs)
    # Local sinks
    if ENABLE_FILE_SINK:
        _write_jsonl(JSONL_PATH, event)
        _write_csv(CSV_PATH, event)
    # Optional sinks
    if es_index:
        _index_to_es(es_client, es_index, event)
    _send_to_api(event)
    logger.info("logged: %s | query='%s' | reward=%s", event_type, event.get("query", ""), event.get("composite_reward"))
    return event


# --- Read env once ---
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_INDEX = os.getenv("ES_INDEX", "rag_index")
CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4.1")
EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-ada-002")
AZURE_OPENAI_EMBED_MODEL=os.getenv("AZURE_OPENAI_EMBED_MODEL", "text-embedding-ada-002")
OPEN_API_TYPE="azure"
AZURE_OPENAI_EMBED_DEPLOYMENT=os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-ada-002")

# Clear conflicting env vars
# for var in ("OPENAI_API_BASE", "OPENAI_BASE_URL", "OPENAI_API_TYPE"):
#     os.environ.pop(var, None)

# Diagnostics
# print("OPENAI_API_BASE:", os.getenv("OPENAI_API_BASE"))
# print("OPENAI_BASE_URL:", os.getenv("OPENAI_BASE_URL"))

# 3) Verify we’re using the correct class (module should be langchain_openai.*)
# print("Class type:", type(AzureOpenAIEmbeddings))
# print("Class module:", AzureOpenAIEmbeddings.__module__)

# Instantiate embeddings
azure_embeddings = AzureOpenAIEmbeddings(
    azure_deployment=EMBED_DEPLOYMENT,
    model=AZURE_OPENAI_EMBED_MODEL,
    azure_endpoint=AZURE_ENDPOINT,
    openai_api_version=AZURE_API_VERSION,
    openai_api_key=AZURE_API_KEY
)

# Test
# print("Instance created:", type(azure_embeddings))
# print("Embedding length:", len(azure_embeddings.embed_query("hello world")))


# Build the Azure chat model for Ragas (do NOT rely on defaults)
ragas_llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),  # <-- your deployment name in Azure
    model=os.getenv("AZURE_OPENAI_RESPONSES_DEPLOYMENT", "gpt-4.1"),   # optional, aligns to your deployment
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0
)


EXCEL_FILE = "dummy_data_output.xlsx"
HIGH_CONFIDENCE = 80
LOW_CONFIDENCE = 50
VERY_LOW_CONFIDENCE = 30

# --- Define clients at the top level ---
try:
    es = Elasticsearch(ES_URL, request_timeout=30)
except Exception as e:
    st.error(f"Elasticsearch connection error: {e}")
    st.stop()

try:
    client = AzureOpenAI(
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
    )
except Exception as e:
    st.error(f"Azure OpenAI client error: {e}")
    st.stop()

# --- State schema ---
class AgentState(TypedDict, total=False):
    query: str
    plan: Optional[str]
    context: Optional[str]
    response: Optional[str]
    response_error: Optional[str]
    scores: Optional[List[float]]
    doc_ids: Optional[List[str]]  # 3.2: carry doc ids


#   
def apply_feedback_to_docs(es_client, index: str, doc_ids: List[str], composite_reward: float):
    """
    Nudges a per-document 'feedback_score' up/down based on composite reward.
    Mapping: reward in [0,1] → delta in [-0.5, +0.5]
    """
    if not doc_ids:
        return
    doc_ids = doc_ids[:20]  # Limit to top 20 docs to avoid long feedback loops
    delta = (composite_reward - 0.5)
    script = {
        "source": """
            if (ctx._source.containsKey('feedback_score')) {
              ctx._source.feedback_score += params.delta;
            } else {
              ctx._source.feedback_score = params.delta;
            }
        """,
        "lang": "painless",
        "params": {"delta": delta}
    }
    for _id in doc_ids:
        try:
            es_client.options(request_timeout=2).update(index=index, id=_id, body={"script": script})
        except Exception:
            # don't break the UX if a single update fails
            pass


def extract_filters_from_query(query):
    """
    Extracts field-value filters from queries like '... with BaseLocation is Mysore'
    Returns a dict: {field: value}
    """
    filters = {}
    # This regex matches 'with Field is Value' or 'where Field is Value'
    matches = re.findall(r'(?:with|where)\s+(\w+)\s+is\s+([A-Za-z0-9_ ]+)', query, re.IGNORECASE)
    for field, value in matches:
        filters[field.strip()] = value.strip()
    return filters

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
    match = re.search(r"Confidence[:\-]?\s*(\d{1,3})\s*%", response, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None

# --- Planner ---
def planner(state: AgentState) -> AgentState:
    query = state.get("query") or ""
    state["plan"] = f"1) Retrieve relevant context\n2) Generate answer for: {query}"
    return state

# --- LLM-based groundedness scorer ---
def score_groundedness_with_llm(answer: str, contexts: List[str]) -> tuple[int, str]:
    """
    Returns (score, rationale). The grader is instructed to rely ONLY on the contexts.
    Robust against parser issues; limits context size; single-line row chunks recommended.
    """
    # Clean and limit contexts to avoid token bloat
    safe_contexts = []
    for c in contexts:
        if isinstance(c, str):
            s = c.strip()
            if s:
                safe_contexts.append(s[:1500])  # hard cap per chunk
    safe_contexts = safe_contexts[:25]  # limit number of chunks

    if not safe_contexts or not (isinstance(answer, str) and answer.strip()):
        return 0, "No answer or context to score."

    # Number the contexts so the grader and generator can reference [c1], [c2], ...
    contexts_block = "\n\n---\n".join([f"[c{i+1}] {c}" for i, c in enumerate(safe_contexts)])

    user_content = (
        "You are a STRICT fact-checker.\n"
        "Score how well the ANSWER is supported ONLY by the CONTEXTS on a 0-100 integer scale (0=no support, 100=fully supported).\n"
        "Return compact JSON: {\"score\": <0-100>, \"why\": \"<=2 sentences\"}\n\n"
        f"CONTEXTS:\n{contexts_block}\n\nANSWER:\n{answer}\n\nJSON:"
    )

    try:
        resp = client.chat.completions.create(
            model=CHAT_DEPLOYMENT,
            messages=[
                {"role": "system",
                 "content": "Evaluate groundedness strictly against the provided contexts. Do NOT use outside knowledge."},
                {"role": "user", "content": user_content},
            ],
            temperature=0
        )
        text = resp.choices[0].message.content or ""
        # Try JSON parse
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            obj = json.loads(m.group(0))
            score = int(obj.get("score", 0))
            why = str(obj.get("why", "")).strip()
            return max(0, min(100, score)), (why or "No rationale")

        # Fallback: first integer in text
        m = re.search(r"\b(\d{1,3})\b", text)
        score = int(m.group(1)) if m else 0
        return max(0, min(100, score)), "Parsed groundedness score heuristically."

    except Exception as e:
        # Keep UI responsive and explicit about errors
        return 0, f"Scoring error: {e}"

# --- LLM-based groundedness scorer ---
def score_groundedness_with_llm2(answer: str, contexts: List[str]) -> tuple[int, str]:
    """
    Returns (score, rationale). The grader is instructed to rely ONLY on the contexts.
    """
    # Trim very large contexts to stay token-safe; keep first 4000 chars each.
    safe_contexts = [c[:4000] for c in contexts if isinstance(c, str) and c.strip()]
    if not safe_contexts:
        return 0, "No context provided to verify grounding."

    # Build a compact scoring prompt; output JSON for robust parsing
    contexts_block = "\n\n---\n".join([f"[c{i+1}] {c}" for i, c in enumerate(safe_contexts)])
    user_content = (
        "You are a STRICT fact-checker.\n"
        "Score how well the ANSWER is supported ONLY by the CONTEXTS on a 0-100 integer scale (0=no support, 100=fully supported).\n"
        "Return compact JSON: {\"score\": <0-100>, \"why\": \"<=2 sentences\"}\n\n"
        f"CONTEXTS:\n{contexts_block}\n\nANSWER:\n{answer}\n\nJSON:"
    )

    try:
        resp = client.chat.completions.create(
            model=CHAT_DEPLOYMENT,
            messages=[
                {"role": "system",
                 "content": "You only evaluate groundedness against the provided contexts; do not use outside knowledge."},
                {"role": "user", "content": user_content},
            ],
            temperature=0
        )
        text = resp.choices[0].message.content or ""
        # Try to parse a JSON object from the response
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            obj = json.loads(m.group(0))
            score = int(obj.get("score", 0))
            why = str(obj.get("why", "")).strip()
            score = max(0, min(100, score))
            return score, (why or "No rationale")
        # Fallback: extract first integer
        m = re.search(r"\b(\d{1,3})\b", text)
        score = int(m.group(1)) if m else 0
        return max(0, min(100, score)), "Parsed groundedness score heuristically."
    except Exception as e:
        return 0, f"Scoring error: {e}"    #Ask the LLM to grade how well the answer is grounded in the provided contexts, 0-100.

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
    """
    Retrieves relevant context from Elasticsearch, cleans empty chunks,
    limits context size, and prepares data for Ragas evaluation.
    """
    query = state.get("query") or ""
    filters = extract_filters_from_query(query)
    must_clauses = []
    for field, value in filters.items():
        must_clauses.append({"match_phrase": {field: value}})
    es_query = {"query": {"bool": {"must": must_clauses}}} if must_clauses else {"query": {"match_all": {}}}
       # Use function_score to include feedback_score for RLHF-lite re-ranking
    body = {
        "query": {
            "function_score": {
                "query": es_query["query"],
                "boost_mode": "sum",
                "score_mode": "sum",
                "functions": [
                    {"field_value_factor": {"field": "feedback_score", "factor": 1.0, "missing": 0.0}}
                ]
            }
        },
        "_source": RELEVANT_FIELDS,
        "size": 50  # Limit ES hits for performance
    }
    try:
        res = es.search(index=ES_INDEX, body=body)
        hits = res.get("hits", {}).get("hits", [])
        contexts, chunks, scores, doc_ids = [], [], [], []
        for h in hits:
            src = h.get("_source", {}) or {}
            doc_id = h.get("_id")
            if doc_id:
                doc_ids.append(doc_id)  # 3.2: collect doc ids
            score = h.get("_score", 0)
            scores.append(score)
            context_parts = [f"{field}: {src[field]}" for field in RELEVANT_FIELDS if field in src]
            context_str = "; ".join(context_parts)
            # Only add non-empty context strings
            if context_str.strip():
                contexts.append(context_str)
            if context_parts:
                chunks.append("; ".join(context_parts))
        
                # ✅ Clean empty contexts
        contexts = [c for c in contexts if c.strip()]
        chunks = [c for c in chunks if c.strip()]

        # ✅ Limit context size for LLM and Ragas
        contexts = contexts[:5]  # Top 5 chunks only
        chunks = chunks[:5]

        # Prepare state
        state["context"] = "\n\n".join(chunks) if chunks else "No ES results matched your filters."
        state["scores"] = scores
        state["doc_ids"] = doc_ids

        # Save for Ragas evaluation
        st.session_state["last_contexts"] = contexts
        st.session_state["last_chunks"] = chunks
        st.session_state["last_doc_ids"] = doc_ids

        # If you have a mapping of queries to ground truth answers
        ground_truth_dict = {
            "Show me the list of employees with BaseLocation is Mysore": "List of employees whose BaseLocation is Mysore.",
            "Show me the list of employees with BaseLocation is Bangalore": "List of employees whose BaseLocation is Bangalore.",
            "Show me the work from home details for 2024": "List of employees and their work from office details.",
            "Show me the list of employees and their role": "List of employees and their roles.",
            "Show me the project details for Q3 FY24": "List of projects and their details for Q3 FY24.",
            "Show me the master data for FY 24": "Master data details for FY 24.",
            "Show me the transfer details where the transfer date is after Jan 2023": "List of employees whose transfer date is after Jan 2023.",
            "show me my leave eligibility": "List of employees and their leave eligibility.",
            "show me the accommodation details which are active": "List of employees and their accommodation details.",
            # Add more mappings as needed
        }

        user_query = st.session_state.get("last_query", "")
        ground_truth = ground_truth_dict.get(user_query, "")

        # Store in session state
        st.session_state["last_ground_truth"] = ground_truth

    except Exception as e:
        state["context"] = f"Retriever error: {e}"
    return state

def generator(state: AgentState) -> AgentState:
    """
    Grounded generator:
    - Uses ONLY the retrieved context (numbered) to answer.
    - Requires inline citations [c1], [c2], ... pointing to the numbered context.
    - Outputs 'Confidence: XX%' and 'Explanation: ...' at the end so your parser continues to work.
    """
    query = state.get("query", "")

    # Prefer the chunked context prepared by the retriever; fall back to state["context"] split
    chunks = st.session_state.get("last_chunks", [])
    if not chunks:
        ctx_str = (state.get("context") or "").strip()
        if ctx_str:
            # Split on double-newline to create pseudo-chunks
            chunks = [c for c in ctx_str.split("\n\n") if isinstance(c, str) and c.strip()]

    # Normalize and trim to keep tokens under control
    processed_chunks: List[str] = []
    for ch in chunks:
        s = str(ch).strip()
        if s:
            processed_chunks.append(s[:2000])  # trim oversized chunks

    # Build numbered context so the model can cite [c1], [c2], ...
    numbered_context = "\n".join(
        [f"[c{i+1}] {ch}" for i, ch in enumerate(processed_chunks)]
    )
    if not numbered_context:
        numbered_context = "NO_CONTEXT"

    system_content = (
        "You are a STRICTLY-GROUNDED assistant. Answer ONLY using the provided CONTEXT.\n"
        "If the answer cannot be found in the context, reply exactly: 'Not found in context.'\n"
        "When stating facts, add in-line citations like [c1], [c2] that reference the numbered context items.\n"
        "At the end of your response add exactly two lines:\n"
        "  1) Confidence: XX%   (computed ONLY from how completely the answer is supported by the CONTEXT)\n"
        "  2) Explanation: ...  (a brief rationale about support from the CONTEXT)"
    )

    user_content = (
        f"QUERY:\n{query}\n\n"
        f"CONTEXT (numbered):\n{numbered_context}\n\n"
        "Write your answer grounded strictly in the context above. "
        "If the context is insufficient, reply 'Not found in context.'"
    )

    try:
        resp = client.chat.completions.create(
            model=CHAT_DEPLOYMENT,  # your Azure deployment name
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            temperature=0.1,
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

# --- Generator via Chat Completions ---
def generator2(state: AgentState) -> AgentState:
    query = state.get("query", "")
    # Use the chunks your retriever assembled earlier and stored in session:
    chunks = st.session_state.get("last_chunks", [])
    # Keep only non-empty chunks and enumerate them for citations [c1], [c2], ...
    numbered_context = "\n".join(
        [f"[c{i+1}] {ch}" for i, ch in enumerate(chunks) if isinstance(ch, str) and ch.strip()]
    )
    if not numbered_context.strip():
        numbered_context = "NO_CONTEXT"


    messages = [
        {
            "role": "system",
            "content": (
                "You are a STRICTLY-GROUNDED assistant. You must answer ONLY using the provided context. "
                "If the answer is not in the context, reply exactly: 'Not found in context.' "
                "When stating a fact, quote the relevant row/cell from the table and cite [c1], [c2], etc. "
                "At the end, add a single line 'Confidence: XX%' reflecting ONLY how completely the answer is supported by the provided context."
            ),
        },
        {
            "role": "user",
            "content": (
                f"QUERY:\n{query}\n\n"
                f"CONTEXT (numbered):\n{numbered_context}\n\n"
                "Write the answer grounded strictly in the context above. If insufficient, say 'Not found in context.'"
            ),
        },
    ]

    try:
        resp = client.chat.completions.create(
            model=CHAT_DEPLOYMENT,
            messages=messages,
            temperature=0.1,
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

# --- Streamlit UI ---
st.set_page_config(page_title="Data Assist Q&A Assistant", layout="wide")
st.title("Data Assist Q&A Assistant")
tab1, tab2, tab3 = st.tabs(["Data Assist Q&A Assistant", "Ragas Evaluation Dashboard", "Table Explorer"])


def prioritize_tables_by_keywords(top_tables, user_query):
    keywords = [kw.lower() for kw in re.findall(r'\w+', user_query)]
    def score_table(table_name):
        return sum(kw in table_name.lower() for kw in keywords)
    return sorted(top_tables, key=lambda t: score_table(t[0]), reverse=True)

# Code copied from backup file start here
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
def build_llm_prompt(user_query, table, df, table_confidence=None, confidence_threshold=4.0):
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
            f"Table selection confidence: {table_confidence}\n"
            "Here is the relevant data from the table (if any):\n"
            #f"{df.head(50).to_markdown(index=False) if df is not None and not df.empty else 'No relevant data.'}\n\n"
            f"{df.to_markdown(index=False) if df is not None and not df.empty else 'No relevant data.'}\n\n"
            "Based on this data and the table selection confidence, answer the user's question in a concise, business-friendly way. "
            "If the table selection confidence is low (e.g., below 5), or the data is insufficient, your confidence score should be low (e.g., 0-30%). "
            "If the data is sufficient and the table selection confidence is high (e.g., above 8), your confidence score can be high (e.g., 80-100%). "
            "After your answer, provide a confidence score (0-100%) for your answer, based on the relevance and completeness of the context and the table selection confidence. Format: \"Confidence: XX%\""
            "If the data is sufficient and the table selection confidence is high (e.g., above 5), your confidence score can be high (e.g., 80-100%). "
            "**Then, provide a brief explanation (3-4 sentences) of why you gave this confidence score, referencing the data or lack thereof. Format: \"Explanation: ...\"**"
            "If the provided data is insufficient, missing, or not relevant to the user's question, your confidence score should be low (e.g., 0-30%). Only give a high confidence score (e.g., 80-100%) if the answer is fully supported by the data."

        )
        return prompt


def extract_markdown_table_2(text):
    """
    Extracts the first Markdown table from the text.
    Returns the table as a string, or None if not found.
    Handles standard Markdown tables with or without trailing newlines.
    """
    # This pattern matches a markdown table anywhere in the text, even at the end
    pattern = (
        r"((?:\|[^\n]+\|\s*\n)"         # header row
        r"(?:\|[ \t:?\-]+?\|\s*\n)"     # separator row (e.g., |---|---|)
        r"(?:\|[^\n]+\|\s*\n?)+)"       # at least one data row, allow optional newline at end
    )
    matches = re.findall(pattern, text)
    if matches:
        # Return the longest match (most likely the real table)
        return max(matches, key=len).strip()

    # Fallback: any block of lines that look like a table (at least 2 rows)
    pattern_simple = r"((?:\|[^\n]+\|\s*\n?){2,})"
    matches = re.findall(pattern_simple, text)
    if matches:
        return max(matches, key=len).strip()

    return None

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
    #st.write("Raw LLM answer:", llm_answer)
    md_table = extract_markdown_table(llm_answer)
    #st.write("Markdown Table:", md_table)
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
# code copied from backup file ends here

from datetime import datetime, timezone
#--- Ragas Data Logging Function ---
def log_ragas_data_es(
    es_client,
    index,
    question,
    answer,
    contexts,
    ground_truth,
    user_feedback=None,
    llm_score=None,
    llm_explanation=None,
    doc_ids: Optional[List[str]] = None,
    faithfulness: Optional[float] = None,
    answer_relevancy: Optional[float] = None
):
    doc = {
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "ground_truth": ground_truth,
        "user_feedback": user_feedback,
        "llm_score": llm_score,
        "llm_explanation": llm_explanation,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "doc_ids": doc_ids or [],
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy
    }
    composite_reward = compute_composite_reward(
        user_feedback=user_feedback,
        llm_score=llm_score,
        faithfulness=faithfulness,
        answer_relevancy=answer_relevancy
    )
    doc["composite_reward"] = composite_reward

    # --- Modular log: ragas_evaluated ---
    try:
        log_event(
            "ragas_evaluated",
            es_client=es_client,
            es_index="event_stream",
            query=question,
            table_confidence=None,
            llm_score=llm_score,
            explanation=llm_explanation,
            user_feedback=user_feedback,
            composite_reward=composite_reward,
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            context_count=len(contexts or []),
            doc_ids=doc_ids or [],
            extra={"phase": "ragas_logging"}
        )
        event=log_event(
            "ragas_evaluated",
            es_client=es_client,
            es_index="event_stream",
            query=question,
            table_confidence=None,
            llm_score=llm_score,
            explanation=llm_explanation,
            user_feedback=user_feedback,
            composite_reward=composite_reward,
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            context_count=len(contexts or []),
            doc_ids=doc_ids or [],
            extra={"phase": "ragas_logging"}
        )
        send_to_aiopslab(event)
    except Exception as _e:
        # Never break caller; we already have ES logging above
        logger.warning("modular ragas log failed: %s", _e)

    try:
        es_client.index(index=index, document=doc)
        st.success("Ragas-format data logged to Elasticsearch!")
        # 3.3 apply feedback to docs
        apply_feedback_to_docs(es_client, ES_INDEX, doc["doc_ids"], composite_reward)
    except Exception as e:
        st.error(f"Failed to log Ragas data to Elasticsearch: {e}")

#--- Fetch all documents from ES for ragas evaluation ---
def fetch_all_for_ragas_from_es(index, es_client, batch_size=100):
    results = []
    page = es_client.search(
        index=index,
        body={"query": {"match_all": {}}},
        scroll='2m',
        size=batch_size
    )
    sid = page['_scroll_id']
    scroll_size = len(page['hits']['hits'])
    while scroll_size > 0:
        for hit in page['hits']['hits']:
            results.append(hit['_source'])
        page = es_client.scroll(scroll_id=sid, scroll='2m')
        sid = page['_scroll_id']
        scroll_size = len(page['hits']['hits'])
    return results

# --- Feedback Section Function ---
def log_user_feedback_es(es_client, index, query, confidence, feedback, llm_score, llm_explanation, doc_ids: Optional[List[str]] = None):
    doc = {
        "query": query,
        "confidence": confidence,
        "feedback": feedback,
        "llm_score": llm_score,
        "llm_explanation": llm_explanation, 
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "doc_ids": doc_ids or []
    }
    composite_reward = compute_composite_reward(
        user_feedback=feedback,
        llm_score=llm_score,
        faithfulness=None,
        answer_relevancy=None
    )
    doc["composite_reward"] = composite_reward
    try:
        es_client.index(index=index, document=doc)
        st.success("Thank you for your feedback! (Logged)")
        apply_feedback_to_docs(es_client, ES_INDEX, doc.get("doc_ids", []), composite_reward)
    except Exception as e:
        st.error(f"Failed to log feedback to Elasticsearch: {e}")

#--- Composite Reward Calculation ---
def compute_composite_reward(user_feedback: str | None,
                             llm_score: float | None,
                             faithfulness: float | None,
                             answer_relevancy: float | None) -> float:
    uf = 1.0 if user_feedback == "thumbs_up" else (0.0 if user_feedback == "thumbs_down" else None)
    ls = (llm_score / 100.0) if llm_score is not None else None
    parts = []
    if uf is not None: parts += [0.4 * uf]
    if ls is not None: parts += [0.2 * ls]
    if faithfulness is not None: parts += [0.2 * float(faithfulness)]
    if answer_relevancy is not None: parts += [0.2 * float(answer_relevancy)]
    return round(sum(parts) / max(1, len(parts)), 4) if parts else 0.0

# --- Sidebar: Health / Telemetry --- 
with st.sidebar:
    st.subheader("Health / Telemetry")
    if Path(LOG_DIR / "app_runtime.log").exists():
        with (LOG_DIR / "app_runtime.log").open("r", encoding="utf-8") as f:
            lines = f.readlines()[-10:]  # tail
        st.text("Recent log tail:")
        st.code("".join(lines), language="text")    
        st.write(f"Events JSONL: {JSONL_PATH}")
    st.write(f"Events CSV:   {CSV_PATH}")

# --- Tab 1: Data Assist Q&A ---
with tab1:
    st.header("Ask a Question and Get Answers from Project Details Data")
    default_query = "List Project Details for Q3"
    user_query = st.text_input("Enter your question:", value="Show me the project details for Q3 FY24")

    if st.button("Submit", key="rag_llm_submit"):
        try:
            result = lg_app.invoke({"query": user_query})
        except Exception as e:
            st.error(f"An error occurred while processing your request: {e}")
            st.stop()
        
        # Enhanced processing with table intelligence
        top_tables = get_top_tables(user_query, top_n=5, min_confidence=2.0)
        top_tables = prioritize_tables_by_keywords(top_tables, user_query)
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
            st.info("No relevant table found for your query.")
            st.stop()
 
        with st.spinner("Processing..."):
            table_name, table_confidence = top_tables[0]
            df, err = filter_excel_data(table_name, filters)

            if err:
                st.error(err)
                st.stop()
            if df is None or df.empty:
                st.warning("No matching data found in Excel for your query and filters.")
                st.stop()
            
            # Show data preview info
            if table_confidence is not None and table_confidence > 2.5:
                st.info("Here is the output data preview based on your query and filters:")     
            elif table_confidence == 0:
                st.info("No relevant data found for your query, so no data preview is shown.")
            else:
                st.info("Data may not be relevant to your query, so preview is hidden.")

            # Build LLM prompt and get answer
            llm_prompt = build_llm_prompt(user_query, table_name, df, table_confidence, confidence_threshold=4.0)
            llm_answer = ask_llm(llm_prompt)
             # Store for llm_answer
            st.session_state["last_llm_answer"] = llm_answer
            
            # # Show enhanced answer with table formatting
            show_llm_answer_table(llm_answer)

            # # Extract confidence and explanation
            # confidence, explanation = extract_confidence_and_explanation(llm_answer)

            #llm_answer = ask_llm(llm_prompt)

            # data = fetch_all_for_ragas_from_es("ragas_eval", es)
            # df = pd.DataFrame(data)
            # dataset = Dataset.from_pandas(df)

            # Build a faithful evaluation context from the table you actually showed
            table_context_for_eval = []
            if df is not None and not df.empty:
                table_context_for_eval = [
                    ' | '.join(map(str, row))
                    for _, row in df.head(50).iterrows()
                ]
            else:
                table_context_for_eval = []

            st.session_state["table_context_for_eval"] = table_context_for_eval

            # >>> Groundedness scoring replaces self-reported confidence <<<
            grounded_score, grounded_why = score_groundedness_with_llm(llm_answer, table_context_for_eval)
            confidence = grounded_score
            explanation = grounded_why

            if confidence == 0 and isinstance(explanation, str) and "Scoring error:" in explanation:
                st.warning(f"Could not compute groundedness score: {explanation}")

            # Also pass the same contexts to Ragas (list[str]) so faithfulness can reflect what the user saw
            st.session_state["last_contexts"] = table_context_for_eval

            # Show LLM prompt for transparency
            if table_confidence is not None and table_confidence > 2.5:
                with st.expander("Show LLM Prompt"):
                    st.code(llm_prompt, language="markdown")
                    
            # Display confidence score with enhanced styling
            if confidence is not None:
                if confidence >= HIGH_CONFIDENCE:
                    st.success(f"**LLM Confidence Score:** {confidence}% (High)")
                    st.success(f"**Why this score?** {explanation}")
                elif confidence >= LOW_CONFIDENCE:
                    st.warning(f"**LLM Confidence Score:** {confidence}% (Medium)")
                    st.warning(f"**Why this score?** {explanation}")
                    st.info("Consider verifying this answer or providing more details for higher accuracy.")
                elif confidence >= VERY_LOW_CONFIDENCE:
                    st.error(f"**LLM Confidence Score:** {confidence}% (Low)")
                    st.error(f"**Why this score?** {explanation}")
                    st.info("This answer may be unreliable. Please rephrase your query or check your data sources.")
                else:
                    st.error(f"**LLM Confidence Score:** {confidence}% (Very Low)")
                    st.error(f"**Why this score?** {explanation}")
                    st.info("No answer shown due to very low confidence. Please provide more details or try a different query.")
            
            # Store for feedback
            st.session_state["last_query"] = user_query
            st.session_state["last_confidence"] = confidence
            st.session_state["last_explanation"] = explanation
            st.session_state["answer_shown"] = True

    # Feedback section (always visible after answer)
    if st.session_state.get("answer_shown", False):
        st.markdown("#### Was this answer helpful?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Good Experience", key="thumbs_up"):
                st.session_state["last_feedback"] = "thumbs_up"
                log_user_feedback_es(
                    es,
                    "user_feedback",
                    st.session_state.get("last_query", ""),
                    st.session_state.get("last_confidence", 0),
                    "thumbs_up",
                    st.session_state.get("last_confidence", 0),  # LLM score
                    st.session_state.get("last_explanation", ""), # LLM explanation
                    doc_ids=st.session_state.get("last_doc_ids", [])  # 3.2→3.3
                )
                # --- Modular log: user_feedback ---
                log_event(
                    "user_feedback",
                    es_client=es,
                    es_index="event_stream",
                    query=st.session_state.get("last_query", ""),
                    table_confidence=None,
                    llm_score=st.session_state.get("last_confidence", 0),
                    explanation=st.session_state.get("last_explanation", ""),
                    user_feedback=st.session_state.get("last_feedback", ""),
                    composite_reward=None,  # you also compute/attach in ES; we can attach here if desired
                    faithfulness=None,
                    answer_relevancy=None,
                    context_count=len(st.session_state.get("table_context_for_eval", [])),
                    doc_ids=st.session_state.get("last_doc_ids", []),
                    extra={"phase": "feedback_clicked"}
                )
                event=log_event(
                    "user_feedback",
                    es_client=es,
                    es_index="event_stream",
                    query=st.session_state.get("last_query", ""),
                    table_confidence=None,
                    llm_score=st.session_state.get("last_confidence", 0),
                    explanation=st.session_state.get("last_explanation", ""),
                    user_feedback=st.session_state.get("last_feedback", ""),
                    composite_reward=None,  # you also compute/attach in ES; we can attach here if desired
                    faithfulness=None,
                    answer_relevancy=None,
                    context_count=len(st.session_state.get("table_context_for_eval", [])),
                    doc_ids=st.session_state.get("last_doc_ids", []),
                    extra={"phase": "feedback_clicked"}
                )
                send_to_aiopslab(event)
                st.success("Feedback logged!")
        with col2:
            if st.button("👎 Bad Retrieval/Experience", key="thumbs_down"):
                st.session_state["last_feedback"] = "thumbs_down"
                log_user_feedback_es(
                    es,
                    "user_feedback",
                    st.session_state.get("last_query", ""),
                    st.session_state.get("last_confidence", 0),
                    "thumbs_down",
                    st.session_state.get("last_confidence", 0),  # LLM score
                    st.session_state.get("last_explanation", ""), # LLM explanation
                    doc_ids=st.session_state.get("last_doc_ids", []) # 3.2→3.3
                )
                # --- Modular log: user_feedback ---
                log_event(
                    "user_feedback",
                    es_client=es,
                    es_index="event_stream",
                    query=st.session_state.get("last_query", ""),
                    table_confidence=None,
                    llm_score=st.session_state.get("last_confidence", 0),
                    explanation=st.session_state.get("last_explanation", ""),
                    user_feedback=st.session_state.get("last_feedback", ""),
                    composite_reward=None,  # you also compute/attach in ES; we can attach here if desired
                    faithfulness=None,
                    answer_relevancy=None,
                    context_count=len(st.session_state.get("table_context_for_eval", [])),
                    doc_ids=st.session_state.get("last_doc_ids", []),
                    extra={"phase": "feedback_clicked"}
                )
                event=log_event(
                    "user_feedback",
                    es_client=es,
                    es_index="event_stream",
                    query=st.session_state.get("last_query", ""),
                    table_confidence=None,
                    llm_score=st.session_state.get("last_confidence", 0),
                    explanation=st.session_state.get("last_explanation", ""),
                    user_feedback=st.session_state.get("last_feedback", ""),
                    composite_reward=None,  # you also compute/attach in ES; we can attach here if desired
                    faithfulness=None,
                    answer_relevancy=None,
                    context_count=len(st.session_state.get("table_context_for_eval", [])),
                    doc_ids=st.session_state.get("last_doc_ids", []),
                    extra={"phase": "feedback_clicked"}
                )
                send_to_aiopslab(event)
                st.success("Feedback logged!")
        # Log Ragas-format data
        log_ragas_data_es(
            es,
            "ragas_eval",
            question=st.session_state.get("last_query", ""),
            answer=st.session_state.get("last_llm_answer", ""),
            contexts=st.session_state.get("table_context_for_eval", []),
            ground_truth=st.session_state.get("last_ground_truth", ""),
            user_feedback=st.session_state.get("last_feedback", ""),
            llm_score=st.session_state.get("last_confidence", 0),
            llm_explanation=st.session_state.get("last_explanation", "")
        )

with tab2:
        st.header("Ragas Evaluation Dashboard")
        st.markdown("This dashboard shows the evaluation of the Q&A assistant using Ragas framework.")
        # Assume results_df has columns: question, faithfulness, answer_relevancy, context_recall
        st.title("Ragas Evaluation Dashboard")
        if st.button("Run Ragas Evaluation"):
            data = fetch_all_for_ragas_from_es("ragas_eval", es)
            df = pd.DataFrame(data)
            df = df.head(50)
            if not df.empty:
                dataset = Dataset.from_pandas(df)

            with st.spinner("Running Ragas evaluation..."):
                try:
                    results = evaluate(
                        dataset,
                        metrics=[
                            faithfulness,
                            answer_relevancy,
                            context_recall
                        ],
                        llm=ragas_llm,
                        embeddings=azure_embeddings
                    )
                    st.success("Ragas evaluation completed!")
                    if results is not None:  # Summary KPIs
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Faithfulness", f"{sum(results['faithfulness'])/len(results['faithfulness']):.2f}")
                        col2.metric("Answer Relevancy", f"{sum(results['answer_relevancy'])/len(results['answer_relevancy']):.2f}")
                        col3.metric("Context Recall", f"{sum(results['context_recall'])/len(results['context_recall']):.2f}")

                        # Trend chart
                        st.subheader("------- Metric Trends Over Time ------")

                        # Convert Ragas scores to DataFrame
                        results_scores = results.scores
                        results_df = pd.DataFrame(results_scores)

                        def color_score(val):
                            if val >= 0.8: return 'background-color: #c8e6c9'  # green
                            elif val >= 0.5: return 'background-color: #fff9c4'  # yellow
                            else: return 'background-color: #ffcdd2'  # red

                        styled_df = results_df.style.map(color_score, subset=['faithfulness','answer_relevancy','context_recall'])

                        st.line_chart(results_df[['faithfulness', 'answer_relevancy', 'context_recall']])

                        # Distribution
                        st.subheader("Score Distribution")
                        st.bar_chart(results_df[['faithfulness', 'answer_relevancy', 'context_recall']])

                        st.subheader("Detailed Results")
                        st.dataframe(styled_df)

                        # Download button
                        csv = results_df.to_csv(index=False)
                        st.download_button("Download Results", csv, "ragas_results.csv", "text/csv")
                    else:
                        st.info("No Ragas evaluation results available yet.")
                except Exception as e:
                    st.error(f"Ragas evaluation failed: {e}")
                results = None

with tab3:
    st.header("Table Explorer")
    try:
        xl = pd.ExcelFile(EXCEL_FILE)
        sheet_names = xl.sheet_names
        selected_sheet = st.selectbox("Select a table (sheet):", sheet_names)
        df = xl.parse(selected_sheet)
        st.write(f"Preview of '{selected_sheet}':")
        st.dataframe(df, width='stretch')
        
        # Search/filter functionality
        search_col = st.selectbox("Filter by column:", df.columns)
        search_val = st.text_input("Filter value:")
        if search_val:
            filtered_df = df[df[search_col].astype(str).str.contains(search_val, case=False, na=False)]
            st.write(f"Filtered preview ({search_col} contains '{search_val}'):")
            st.dataframe(filtered_df, width='stretch')
    except Exception as e:
        st.error(f"Error loading Excel file: {e}")
        try:
            st.write("Available sheets:", xl.sheet_names)
        except:
            st.write("Could not load Excel file information.")
