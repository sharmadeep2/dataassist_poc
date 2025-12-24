import os
import random
import json
from elasticsearch import Elasticsearch
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# Setup
es = Elasticsearch(os.getenv("ES_URL", "http://localhost:9200"))
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)
EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")
ES_INDEX = os.getenv("ES_INDEX", "rag_index")

# Load schema
with open("table_schema_updated_with_descriptions.json", "r", encoding="utf-8") as f:
    schema = json.load(f)

def get_project_details_sample(i):
    # Generate realistic dummy data for Project Details table
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fiscal_years = ["FY23", "FY24", "FY25"]
    quarters = ["FY23 - Q1", "FY23 - Q2", "FY23 - Q3", "FY23 - Q4", "FY24 - Q1", "FY24 - Q2", "FY24 - Q3", "FY24 - Q4"]
    return {
        "Project Details[Master Customer Code]": f"CUST{1000 + i}",
        "Project Details[Service Line Unit]": random.choice(["IT Services", "Consulting", "Finance", "HR"]),
        "Project Details[Service Line Unit 2]": random.choice(["Advisory", "Delivery", "Support"]),
        "Project Details[Project PU]": random.choice(["Unit A", "Unit B", "Unit C"]),
        "Master Project Code": f"PROJ{2000 + i}",
        "Project Details[Revenue Credit Unit]": random.choice(["USD", "EUR", "INR"]),
        "Project Details[Contract Type]": random.choice(["Fixed-price", "Time and Materials", "Cost Plus"]),
        "'Period'[Month]": random.choice(months),
        "'Period'[txtFiscalYear]": random.choice(fiscal_years),
        "'Period'[Fiscal Quarter]": random.choice(quarters),
        "sales_amount": random.randint(50000, 500000),
        "table_name": "Project Details"
    }

def get_sample_value(col, i):
    col_name = col.get('column_name', col.get('name', 'Field'))
    dtype = col.get("dtype", col.get("type", "")).lower()
    # Use sample values if available
    if "sample_values" in col and col["sample_values"]:
        return random.choice([v for v in col["sample_values"] if v not in [None, "null", ""]])
    # Fallbacks based on type
    if "int" in dtype:
        return random.randint(1, 100)
    elif "double" in dtype or "float" in dtype or "real" in dtype:
        return round(random.uniform(1, 100), 2)
    elif "date" in dtype or "datetime" in dtype:
        return "2024-01-01"
    elif "string" in dtype or "char" in dtype or "text" in dtype:
        return f"Sample {col_name}"
    else:
        return f"Sample {col_name}"

for table in schema:
    table_name = table.get("table") or table.get("table_name")
    columns = table.get("schema", table.get("columns", []))
    # For Project Details, ingest 10 realistic dummy rows
    if table_name == "Project Details":
        for i in range(10):
            doc = get_project_details_sample(i)
            content = (
                f"Project {doc['Master Project Code']} for customer {doc['Project Details[Master Customer Code]']} in "
                f"{doc['\'Period\'[Fiscal Quarter]']} ({doc['\'Period\'[Month]']}, {doc['\'Period\'[txtFiscalYear]']}) "
                f"managed by {doc['Project Details[Service Line Unit]']} generated {doc['sales_amount']} {doc['Project Details[Revenue Credit Unit]']}. "
                f"Contract type: {doc['Project Details[Contract Type]']}. Project PU: {doc['Project Details[Project PU]']}."
            )
            embedding_response = client.embeddings.create(
                input=content,
                model=EMBED_DEPLOYMENT
            )
            embedding = embedding_response.data[0].embedding
            doc["content"] = content
            doc["embedding"] = embedding
            es.index(index=ES_INDEX, document=doc)
            print(f"Indexed sample document {i+1} for table: {table_name}")
    else:
        # For other tables, ingest 1 dummy row as before
        doc = {}
        for col in columns:
            col_name = col.get("column_name") or col.get("name")
            if not col_name:
                continue
            doc[col_name] = get_sample_value(col, 0)
        doc["table_name"] = table_name
        content = f"Sample data for table {table_name}: " + ", ".join([f"{k}={v}" for k, v in doc.items()])
        embedding_response = client.embeddings.create(
            input=content,
            model=EMBED_DEPLOYMENT
        )
        embedding = embedding_response.data[0].embedding
        doc["content"] = content
        doc["embedding"] = embedding
        es.index(index=ES_INDEX, document=doc)
        print(f"Indexed sample document for table: {table_name}")

print("Sample data for all tables indexed!")