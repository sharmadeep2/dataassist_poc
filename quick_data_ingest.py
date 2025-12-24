#!/usr/bin/env python3
"""
Quick data ingestion script for DataAssist POC
Adds sample project data to Elasticsearch for testing
"""

import os
import json
from datetime import datetime
from elasticsearch import Elasticsearch
from dotenv import load_dotenv

load_dotenv()

# Configuration
ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_INDEX = os.getenv("ES_INDEX", "rag_index")

# Initialize Elasticsearch client
try:
    es = Elasticsearch(
        ES_URL,
        request_timeout=30,
        verify_certs=False,
        headers={"Accept": "application/vnd.elasticsearch+json; compatible-with=8"}
    )
except Exception as e:
    print(f"Error initializing Elasticsearch client: {e}")
    # Try without headers
    es = Elasticsearch(ES_URL, request_timeout=30, verify_certs=False)

# Sample project data
sample_data = [
    {
        "project_id": "PROJ001",
        "project_name": "Data Analytics Platform",
        "quarter": "Q3",
        "fiscal_year": "FY24",
        "start_date": "2023-10-01",
        "end_date": "2023-12-31",
        "status": "Completed",
        "budget": 500000,
        "team_size": 12,
        "description": "Implementation of comprehensive data analytics platform using Azure services",
        "technologies": ["Azure Data Factory", "Azure Synapse", "Power BI", "Python"],
        "category": "Data & Analytics"
    },
    {
        "project_id": "PROJ002", 
        "project_name": "AI Customer Service Bot",
        "quarter": "Q3",
        "fiscal_year": "FY24",
        "start_date": "2023-10-15",
        "end_date": "2023-12-20",
        "status": "In Progress",
        "budget": 300000,
        "team_size": 8,
        "description": "Development of AI-powered customer service chatbot using Azure OpenAI",
        "technologies": ["Azure OpenAI", "Azure Bot Framework", "Cosmos DB", "JavaScript"],
        "category": "AI & ML"
    },
    {
        "project_id": "PROJ003",
        "project_name": "Cloud Migration Initiative",
        "quarter": "Q4",
        "fiscal_year": "FY24",
        "start_date": "2024-01-01",
        "end_date": "2024-03-31",
        "status": "Planning",
        "budget": 750000,
        "team_size": 15,
        "description": "Migration of legacy applications to Azure cloud infrastructure",
        "technologies": ["Azure App Service", "Azure SQL Database", "Azure DevOps", "Docker"],
        "category": "Infrastructure"
    },
    {
        "project_id": "PROJ004",
        "project_name": "Mobile App Development",
        "quarter": "Q2",
        "fiscal_year": "FY24",
        "start_date": "2023-07-01",
        "end_date": "2023-09-30",
        "status": "Completed",
        "budget": 400000,
        "team_size": 10,
        "description": "Cross-platform mobile application for customer engagement",
        "technologies": ["React Native", "Azure Mobile Apps", "Azure Notification Hubs", "TypeScript"],
        "category": "Mobile Development"
    },
    {
        "project_id": "PROJ005",
        "project_name": "Security Enhancement Program",
        "quarter": "Q3",
        "fiscal_year": "FY24",
        "start_date": "2023-10-01",
        "end_date": "2024-01-31",
        "status": "In Progress",
        "budget": 200000,
        "team_size": 6,
        "description": "Comprehensive security audit and enhancement across all systems",
        "technologies": ["Azure Security Center", "Azure Sentinel", "Azure Key Vault", "PowerShell"],
        "category": "Security"
    }
]

def ingest_data():
    """Ingest sample data into Elasticsearch"""
    try:
        # Test connection
        if not es.ping():
            print("❌ Could not connect to Elasticsearch")
            return False
            
        print(f"✅ Connected to Elasticsearch at {ES_URL}")
        
        # Check if index exists, create if not
        if not es.indices.exists(index=ES_INDEX):
            es.indices.create(index=ES_INDEX)
            print(f"✅ Created index: {ES_INDEX}")
        
        # Ingest documents
        success_count = 0
        for i, doc in enumerate(sample_data):
            try:
                doc["ingestion_timestamp"] = datetime.now().isoformat()
                doc["document_id"] = f"doc_{doc['project_id']}"
                
                response = es.index(
                    index=ES_INDEX, 
                    id=doc["project_id"],
                    document=doc
                )
                
                if response.get('result') in ['created', 'updated']:
                    success_count += 1
                    print(f"✅ Indexed document: {doc['project_name']} ({doc['project_id']})")
                else:
                    print(f"⚠️ Unexpected response for {doc['project_id']}: {response}")
                    
            except Exception as e:
                print(f"❌ Failed to index {doc['project_id']}: {str(e)}")
        
        # Refresh index to make documents searchable
        es.indices.refresh(index=ES_INDEX)
        
        print(f"\n🎉 Successfully indexed {success_count}/{len(sample_data)} documents")
        print(f"📊 Index: {ES_INDEX}")
        print(f"🔍 You can now search and query this data in your DataAssist application!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during data ingestion: {str(e)}")
        return False

def verify_data():
    """Verify the ingested data"""
    try:
        # Get document count
        count_response = es.count(index=ES_INDEX)
        doc_count = count_response['count']
        
        # Get a sample document
        search_response = es.search(
            index=ES_INDEX,
            body={"query": {"match_all": {}}, "size": 1}
        )
        
        print(f"\n📈 Verification Results:")
        print(f"   Total documents: {doc_count}")
        
        if search_response['hits']['hits']:
            sample_doc = search_response['hits']['hits'][0]['_source']
            print(f"   Sample document: {sample_doc.get('project_name', 'Unknown')}")
            print(f"   Categories available: {set([doc['category'] for doc in sample_data])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during verification: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Starting DataAssist POC data ingestion...")
    
    if ingest_data():
        verify_data()
        print("\n✨ Data ingestion complete! Your DataAssist application is ready to use.")
    else:
        print("\n💥 Data ingestion failed. Please check your Elasticsearch connection.")