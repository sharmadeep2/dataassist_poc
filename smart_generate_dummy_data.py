import json
import random
import pandas as pd
from faker import Faker

fake = Faker()

# Load the schema file
with open("table_schema_updated_with_descriptions.json", "r", encoding="utf-8") as f:
    schema = json.load(f)

def smart_sample_value(col, idx):
    # Use sample values if available and cycle through them
    if "sample_values" in col and col["sample_values"]:
        values = [v for v in col["sample_values"] if v not in [None, "null", ""]]
        if values:
            return values[idx % len(values)]
    dtype = col.get("dtype", col.get("data_type", "")).lower()
    name = col.get('column_name', col.get('name', 'Field')).lower()
    # Smart logic for common field types
    if "email" in name:
        return fake.email()
    if "phone" in name or "contact" in name:
        return fake.phone_number()
    if "name" in name and "project" not in name:
        return fake.name()
    if "city" in name or "location" in name:
        return fake.city()
    if "country" in name:
        return fake.country()
    if "code" in name or "id" in name:
        return f"{name[:4].upper()}{random.randint(1000,9999)}"
    if "date" in dtype or "datetime" in dtype:
        return fake.date_time_this_decade().strftime("%Y-%m-%d %H:%M:%S")
    if "int" in dtype:
        return random.randint(1, 10000)
    if "double" in dtype or "float" in dtype or "decimal" in dtype:
        return round(random.uniform(1, 10000), 2)
    if "string" in dtype or "char" in dtype or "text" in dtype:
        # For known categories
        if "month" in name:
            return random.choice(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
        if "quarter" in name:
            return random.choice(["Q1", "Q2", "Q3", "Q4"])
        if "year" in name:
            return random.choice(["FY21", "FY22", "FY23", "FY24", "FY25"])
        if "unit" in name:
            return random.choice(["IT Services", "Finance", "HR", "Consulting", "Operations"])
        if "type" in name:
            return random.choice(["Permanent", "Temporary", "Contract", "Fixed-price", "Time and Materials"])
        if "status" in name:
            return random.choice(["Open", "Closed", "Active", "Inactive", "Pending", "Approved", "Rejected"])
        if "currency" in name or "unit" in name:
            return random.choice(["USD", "EUR", "INR"])
        # Otherwise, generate a plausible string
        return fake.word().capitalize() + f"_{idx+1}"
    # Fallback
    return fake.word().capitalize() + f"_{idx+1}"

excel_writer = pd.ExcelWriter("dummy_data_output.xlsx", engine="xlsxwriter")

for table in schema:
    table_name = table.get("table") or table.get("table_name")
    columns = []
    for col in table.get("schema", table.get("columns", [])):
        col_name = col.get("column_name") or col.get("name") or col.get("measure") or col.get("measure_name")
        if col_name:
            columns.append((col_name, col))
    if not columns:
        continue
    records = []
    for i in range(30):
        row = {}
        for col_name, col in columns:
            row[col_name] = smart_sample_value(col, i)
        records.append(row)
    df = pd.DataFrame(records)
    df.to_excel(excel_writer, sheet_name=table_name[:31], index=False)

excel_writer.close()
print("Smart dummy data for all tables has been written to dummy_data_output.xlsx")
