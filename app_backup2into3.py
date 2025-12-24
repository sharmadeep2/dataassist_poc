import os
import streamlit as st
import pandas as pd
from elasticsearch import Elasticsearch
import openai
import re
from dotenv import load_dotenv

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
# --- CONFIGURATION ---
ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_INDEX = os.getenv("ES_INDEX", "rag_index")
EXCEL_FILE = "dummy_data_output.xlsx"


# --- Initialize clients ---
es = Elasticsearch(ES_URL)

st.set_page_config(page_title="RAG + Excel LLM Demo", layout="wide")
st.title("RAG + Excel LLM Demo")

tab1, tab2 = st.tabs(["Ask LLM (RAG + Excel)", "Table Explorer"])

# --- Tab 1: RAG + Excel + LLM ---
with tab1:
    st.header("Ask a Question (RAG + Excel + LLM)")
    user_query = st.text_input("Enter your question:", value="Show me the project details for Q3 FY24")

    def get_top_tables(user_query, top_n=3):
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
                confidence = hit["_source"]["metadata"].get("confidence", None)
                tables.append((table_name, confidence))
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
        # Load Excel and find best matching sheet
        xl = pd.ExcelFile(EXCEL_FILE)
        sheet_names = xl.sheet_names

        # Normalize for matching
        norm_table = table.replace(" ", "").replace("_", "").lower()
        best_sheet = None
        for s in sheet_names:
            norm_sheet = s.replace(" ", "").replace("_", "").lower()
            if norm_table == norm_sheet:
                best_sheet = s
                break
        if not best_sheet:
            # Try partial match
            for s in sheet_names:
                norm_sheet = s.replace(" ", "").replace("_", "").lower()
                if norm_table in norm_sheet or norm_sheet in norm_table:
                    best_sheet = s
                    break
        if not best_sheet:
            st.warning(f"No matching sheet found for '{table}'. Available sheets: {sheet_names}")
            return None, f"No matching sheet found for '{table}'."

        df = xl.parse(best_sheet)
#        st.write("Columns in selected table:", list(df.columns))
#        st.write("Sample data:", df.head(5))

        # Fuzzy column matching
        fy_col = None
        q_col = None
        for col in df.columns:
            if "year" in col.lower() or "fiscal" in col.lower():
                fy_col = col
            if "quarter" in col.lower() or "qtr" in col.lower():
                q_col = col

        # Apply filters if columns found
        if fy_col and filters["FY"]:
            fy_val = filters["FY"].replace("FY", "").replace(" ", "")
            # Try to match FY24 to 2024, FY 2024, 24, etc.
            df = df[
                df[fy_col].astype(str).str.contains(fy_val, case=False, na=False) |
                df[fy_col].astype(str).str.contains(filters["FY"], case=False, na=False)
            ]
        if q_col and filters["Q"]:
            q_val = filters["Q"].replace("Q", "").replace(" ", "")
            # Try to match Q3 to 3, Qtr3, Quarter 3, etc.
            df = df[
                df[q_col].astype(str).str.contains(q_val, case=False, na=False) |
                df[q_col].astype(str).str.contains(filters["Q"], case=False, na=False)
            ]
        return df, None


    def build_llm_prompt(user_query, table, df):
        preview_df = df.head(10)
        table_str = preview_df.to_markdown(index=False)
        prompt = (
            f"User asked: {user_query}\n"
            f"Here is the relevant data from the '{table}' table:\n\n"
            f"{table_str}\n\n"
            "Based on this data, answer the user's question in a concise, business-friendly way. "
            "If the data is insufficient, say so clearly."
        )
        return prompt

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

    if st.button("Submit", key="rag_llm_submit"):
        with st.spinner("Processing..."):
            # Step 1: Query Elasticsearch for top tables
            top_tables = get_top_tables(user_query)
            if not top_tables:
                st.error("No relevant tables found in Elasticsearch.")
                st.stop()
#            st.write("Top table(s) selected (with confidence):")
#            for t, c in top_tables:
#                st.write(f"- {t} (Confidence: {c if c is not None else 'N/A'})")

            # Step 2: Planner - extract filters
            filters = extract_filters(user_query)
            st.write(f"Filters extracted: {filters}")

            # Step 3: Generator - load and filter Excel data
            df, err = filter_excel_data(top_tables[0][0], filters)
            if err:
                st.error(err)
                st.stop()
            if df is None or df.empty:
                st.warning("No matching data found in Excel for your query and filters.")
                st.stop()
            st.write("Filtered data preview:")
            st.dataframe(df.head(10))

            # Step 4: Build LLM prompt and get answer
            llm_prompt = build_llm_prompt(user_query, top_tables[0][0], df)
            llm_answer = ask_llm(llm_prompt)

            # Step 5: Show answer
            st.subheader("LLM Answer")
            st.write(llm_answer)

            # Optionally, show the LLM prompt for transparency
            with st.expander("Show LLM Prompt"):
                st.code(llm_prompt, language="markdown")

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