import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from openai import AzureOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional

# Define the state structure
class AgentState(TypedDict, total=False):
    query: str
    plan: Optional[str]
    context: Optional[str]
    response: Optional[str]

# Load environment variables
load_dotenv()

# Initialize clients
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

es = Elasticsearch("http://localhost:9200")

# Define the planner node
def planner(state: AgentState) -> AgentState:
    query = state.get("query")
    if not query:
        print("Warning: 'query' key is missing in state")
        return state
    print(f"Planner received query: {query}")
    state["plan"] = f"Plan for: {query}"
    return state

# (Optional) Retriever Agent - currently not used in the graph
def retriever(state: AgentState) -> AgentState:
    query = state.get("query")
    if not query:
        print("Warning: 'query' key is missing in state")
        return state
    embedding = client.embeddings.create(
        input=[query],
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT")
    )
    # You would add code here to use the embedding with Elasticsearch
    # and update state["context"] with retrieved context.
    return state

# (Optional) Generator Agent - currently not used in the graph
def generator(state: AgentState) -> AgentState:
    query = state.get("query")
    context = state.get("context", "")
    if not query:
        print("Warning: 'query' key is missing in state")
        return state
    prompt = f"Answer the following question using the context below:\n\nContext:\n{context}\n\nQuestion:\n{query}"
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    state["response"] = response.choices[0].message.content
    return state

# Build the graph
workflow = StateGraph(AgentState)
workflow.add_node("planner", planner)
workflow.set_entry_point("planner")
workflow.add_edge("planner", END)

# Compile the graph
app = workflow.compile()

if __name__ == "__main__":
    query_input = "Generate sales report for Q3"
    result = app.invoke({"query": query_input})
