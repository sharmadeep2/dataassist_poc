import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from openai import AzureOpenAI, BadRequestError, NotFoundError
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional

# =========================
# State definition
# =========================
class AgentState(TypedDict, total=False):
    query: str
    plan: Optional[str]
    context: Optional[str]
    response: Optional[str]
    response_error: Optional[str]

# =========================
# Config / Clients
# =========================
load_dotenv()

AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

# IMPORTANT: These must be your **Azure deployment names**, not base model names.
CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")      # e.g., "gpt-4o-mini"
EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")    # e.g., "text-embedding-3-large"

ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_INDEX = os.getenv("ES_INDEX", "documents")

client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
)

es = Elasticsearch(ES_URL, request_timeout=30)

# =========================
# Utilities
# =========================
def validate_config():
    problems = []
    if not AZURE_API_KEY:
        problems.append("AZURE_OPENAI_API_KEY is missing")
    if not AZURE_ENDPOINT:
        problems.append("AZURE_OPENAI_ENDPOINT is missing (e.g., https://<resource>.openai.azure.com)")
    if not CHAT_DEPLOYMENT:
        problems.append("AZURE_OPENAI_CHAT_DEPLOYMENT is missing (must be your Azure deployment name)")
    if problems:
        print("⚠️ Configuration issues:\n - " + "\n - ".join(problems))
        print("Please fix your .env and rerun.")
    else:
        print("✅ Azure config looks present. Using:")
        print(f"  - AZURE_OPENAI_API_VERSION = {AZURE_API_VERSION}")
        print(f"  - CHAT_DEPLOYMENT          = {CHAT_DEPLOYMENT}")
        if EMBED_DEPLOYMENT:
            print(f"  - EMBED_DEPLOYMENT         = {EMBED_DEPLOYMENT}")
        else:
            print("  - EMBED_DEPLOYMENT         = (not set; retriever will simulate context)")

# =========================
# Nodes
# =========================
def planner(state: AgentState) -> AgentState:
    query = state.get("query")
    if not query:
        print("Warning: 'query' missing in state.")
        return state
    plan = f"1) Retrieve relevant context\n2) Generate answer for: {query}"
    print("Planner plan:", plan)
    state["plan"] = plan
    return state

def retriever(state: AgentState) -> AgentState:
    query = state.get("query")
    if not query:
        print("Warning: 'query' missing in retriever.")
        return state

    if not EMBED_DEPLOYMENT:
        print("Retriever: No embedding deployment configured; using simulated context.")
        state["context"] = f"Simulated context for query: {query}"
        return state

    try:
        emb_resp = client.embeddings.create(
            input=query,                 # single string is fine
            model=EMBED_DEPLOYMENT       # MUST be your embeddings DEPLOYMENT name
        )
        emb = emb_resp.data[0].embedding

        # Try ES kNN if reachable; otherwise simulate context
        if es.ping():
            # NOTE: Your ES index must have a dense_vector field named "embedding"
            # with the same dimension as the above embedding.
            body = {
                "query": {
                    "bool": {
                        "must": [
                            {"match": {"'Period'[Fiscal Quarter]": "FY24 - Q3"}}
                            ]
                        }
                },
                "knn": {
                    "field": "embedding",
                    "query_vector": emb,
                    "k": 5,
                    "num_candidates": 100
                },
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
                    "'Period'[Fiscal Quarter]"
                ]
            }
            res = es.search(index=ES_INDEX, body=body)
            hits = res.get("hits", {}).get("hits", [])
            chunks = []
            for h in hits:
                src = h.get("_source", {}) or {}
                context_parts = []
                for field in body["_source"]:
                    if field in src:
                        context_parts.append(f"{field}: {src[field]}")
                context = "; ".join(context_parts)
                chunks.append(context)
            state["context"] = "\n\n".join(chunks[:5]) if chunks else f"No relevant data found for: {query}"
        else:
            print("Retriever: Elasticsearch not reachable; using simulated context.")
            state["context"] = f"Simulated context for query: {query}"

    except NotFoundError as e:
        # Raised when the deployment name is wrong or not found
        print("Retriever error: Embeddings deployment not found.")
        print("Hint: Ensure AZURE_OPENAI_EMBED_DEPLOYMENT matches a deployed Embeddings model in Azure.")
        state["context"] = f"Simulated context for query: {query}"
    except BadRequestError as e:
        # Often includes helpful JSON details
        print(f"Retriever BadRequest: {e}")
        print("Hint: Verify your embeddings deployment name and API version.")
        state["context"] = f"Simulated context for query: {query}"
    except Exception as e:
        print("Retriever error:", e)
        state["context"] = f"Simulated context for query: {query}"

    return state

def generator(state: AgentState) -> AgentState:
    query = state.get("query", "")
    context = state.get("context", "")
    prompt = (
        "Use the provided context to answer the question. "
        "If the context is insufficient, say so explicitly and outline what would be needed.\n\n"
        f"Context:\n{context}\n\nQuestion:\n{query}"
    )

    try:
        # Use Chat Completions ONLY, since your SDK lacks the Responses client.
        # IMPORTANT: model must be the Azure deployment name, not a base model name.
        resp = client.chat.completions.create(
            model=CHAT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are a concise, accurate assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        state["response"] = resp.choices[0].message.content

    except BadRequestError as e:
        msg = str(e)
        # If your Azure resource is configured to require the Responses API,
        # Azure may return a confusing "'input' is a required property".
        if "input is a required property" in msg.lower():
            state["response_error"] = (
                "Your Azure resource/API version appears to expect the Responses API "
                "('input' field). Since your SDK doesn't have client.responses, either:\n"
                "  1) Upgrade the 'openai' Python package: pip install -U openai>=1.44\n"
                "  2) Or set AZURE_OPENAI_API_VERSION to one that supports Chat Completions (e.g., 2024-02-15)\n"
                "  3) Or switch your code to call the Responses API via REST.\n"
                "Then retry."
            )
        else:
            state["response_error"] = f"Generation failed (BadRequest): {msg}"
        print(state["response_error"])
    except Exception as e:
        state["response_error"] = f"Generation failed: {e}"
        print(state["response_error"])

    return state

# =========================
# Graph
# =========================
workflow = StateGraph(AgentState)
workflow.add_node("planner", planner)
workflow.add_node("retriever", retriever)
workflow.add_node("generator", generator)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "retriever")
workflow.add_edge("retriever", "generator")
workflow.add_edge("generator", END)

app = workflow.compile()

# =========================
# Run
# =========================
if __name__ == "__main__":
    validate_config()
    query_input = os.getenv("QUERY", "List Project Details for Q3")
    result = app.invoke({"query": query_input})

print("\n--- FINAL STATE ---")
for k in ("plan", "response", "response_error"):
    if k in result:
        print(f"{k}:\n{result[k]}\n")

