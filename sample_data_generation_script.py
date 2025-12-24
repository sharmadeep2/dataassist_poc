import json
import random
import pandas as pd

# Load the schema file
with open("table_schema_updated_with_descriptions.json", "r", encoding="utf-8") as f:
    schema = json.load(f)

def get_sample_value(col, idx):
    # Use sample values if available
    if "sample_values" in col and col["sample_values"]:
        values = [v for v in col["sample_values"] if v not in [None, "null", ""]]
        if values:
            return values[idx % len(values)]
    dtype = col.get("dtype", col.get("data_type", "")).lower()
    # Fallbacks based on type
    if "int" in dtype:
        return idx * 7 + random.randint(1, 100)
    elif "double" in dtype or "float" in dtype or "decimal" in dtype:
        return round(idx * 3.14 + random.uniform(1, 100), 2)
    elif "date" in dtype or "datetime" in dtype:
        year = random.choice([2022, 2023, 2024, 2025])
        month = (idx % 12) + 1
        day = (idx % 28) + 1
        return f"{year}-{month:02d}-{day:02d} 00:00:00"
    elif "string" in dtype or "char" in dtype or "text" in dtype:
        base = col.get('column_name', col.get('name', 'Field'))
        return f"{base}_val_{idx+1}"
    else:
        base = col.get('column_name', col.get('name', 'Field'))
        return f"{base}_val_{idx+1}"

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
            row[col_name] = get_sample_value(col, i)
        records.append(row)
    df = pd.DataFrame(records)
    safe_sheet_name = "".join([c if c.isalnum() else "_" for c in table_name])[:31]
    df.to_excel(excel_writer, sheet_name=safe_sheet_name, index=False)

excel_writer.close()
print("Dummy data for all tables has been written to dummy_data_output.xlsx")