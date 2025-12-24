import streamlit as st
from elasticsearch import Elasticsearch
from datetime import datetime

es = Elasticsearch("http://localhost:9200")

st.write("ES client info:", es.info())

if st.button("Test Feedback Logging"):
    doc = {
        "query": "test query",
        "confidence": 99,
        "feedback": "thumbs_up",
        "timestamp": datetime.utcnow().isoformat()
    }
    try:
        res = es.index(index="user_feedback", document=doc)
        st.success("Test feedback logged!")
    except Exception as e:
        st.error(f"Failed to log feedback to Elasticsearch: {e}")