import os
import random
from elasticsearch import Elasticsearch
from openai import AzureOpenAI
from dotenv import load_dotenv
load_dotenv()


# Setup
es = Elasticsearch("http://localhost:9200")
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)
EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")
ES_INDEX = "rag_index"

# Dummy data options
service_line_units = ["IT Services", "Human Resources", "Marketing", "Finance", "Operations"]
project_pus = ["Unit A", "Unit B", "Unit C", "Unit D"]
contract_types = ["Fixed-price", "Time and Materials", "Cost Plus"]
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
fiscal_years = ["FY23", "FY24", "FY25"]
fiscal_quarters = ["FY23 - Q1", "FY23 - Q2", "FY23 - Q3", "FY23 - Q4", "FY24 - Q1", "FY24 - Q2", "FY24 - Q3", "FY24 - Q4", "FY25 - Q1", "FY25 - Q2", "FY25 - Q3", "FY25 - Q4"]
revenue_units = ["USD", "EUR", "INR"]

for i in range(1, 51):
    master_customer_code = f"CUST{1000 + i}"
    master_project_code = f"PROJ{2000 + i}"
    service_line_unit = random.choice(service_line_units)
    service_line_unit2 = random.choice(service_line_units)
    project_pu = random.choice(project_pus)
    revenue_credit_unit = random.choice(revenue_units)
    contract_type = random.choice(contract_types)
    month = random.choice(months)
    fiscal_year = random.choice(fiscal_years)
    fiscal_quarter = random.choice(fiscal_quarters)
    sales_amount = random.randint(50000, 500000)

    content = (
        f"Project {master_project_code} for customer {master_customer_code} in {fiscal_quarter} "
        f"({month}, {fiscal_year}) managed by {service_line_unit} generated {sales_amount} {revenue_credit_unit}. "
        f"Contract type: {contract_type}. Project PU: {project_pu}."
    )

    # Generate embedding for the content field
    embedding_response = client.embeddings.create(
        input=content,
        model=EMBED_DEPLOYMENT
    )
    embedding = embedding_response.data[0].embedding

    # Build document
    doc = {
        "Project Details[Master Customer Code]": master_customer_code,
        "Project Details[Service Line Unit]": service_line_unit,
        "Project Details[Service Line Unit 2]": service_line_unit2,
        "Project Details[Project PU]": project_pu,
        "Master Project Code": master_project_code,
        "Project Details[Revenue Credit Unit]": revenue_credit_unit,
        "Project Details[Contract Type]": contract_type,
        "'Period'[Month]": month,
        "'Period'[txtFiscalYear]": fiscal_year,
        "'Period'[Fiscal Quarter]": fiscal_quarter,
        "sales_amount": sales_amount,
        "content": content,
        "embedding": embedding
    }

    # Index the document
    es.index(index=ES_INDEX, document=doc)
    print(f"Indexed document {i}: {master_project_code}")

print("All 50 dummy project details indexed!")