import pandas as pd
from elasticsearch import Elasticsearch
import os

# Set your Elasticsearch endpoint and index name
ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_INDEX = os.getenv("ES_INDEX", "rag_index")
EXCEL_FILE = "dummy_data_output.xlsx"

es = Elasticsearch(ES_URL)

# Load the Excel file
xl = pd.ExcelFile(EXCEL_FILE)
sheet_names = xl.sheet_names

for sheet in sheet_names:
    df = xl.parse(sheet)
    df = df.fillna("")  # Replace NaN with empty string for ES compatibility
    for _, row in df.iterrows():
        doc = row.to_dict()
        doc["table_name"] = sheet  # Add the sheet name as table_name
        es.index(index=ES_INDEX, document=doc)
    print(f"Ingested {len(df)} records from sheet '{sheet}'.")

print("All Excel data has been ingested into Elasticsearch!")