#!/usr/bin/env python3
"""
Simple data ingestion using requests library
Works around Elasticsearch client version compatibility issues
"""

import os
import json
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Configuration
ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_INDEX = os.getenv("ES_INDEX", "rag_index")

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
        "description": "Implementation of comprehensive data analytics platform using Azure services for Q3 FY24",
        "technologies": ["Azure Data Factory", "Azure Synapse", "Power BI", "Python"],
        "category": "Data & Analytics",
        "table_name": "project_details"
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
        "description": "Development of AI-powered customer service chatbot using Azure OpenAI for Q3 FY24",
        "technologies": ["Azure OpenAI", "Azure Bot Framework", "Cosmos DB", "JavaScript"],
        "category": "AI & ML",
        "table_name": "project_details"
    },
    {
        "project_id": "PROJ003",
        "project_name": "Q3 Financial Analysis",
        "quarter": "Q3",
        "fiscal_year": "FY24", 
        "start_date": "2023-10-01",
        "end_date": "2023-12-31",
        "status": "Completed",
        "budget": 150000,
        "team_size": 5,
        "description": "Quarterly financial analysis and reporting for Q3 FY24 with advanced analytics",
        "technologies": ["Excel", "Power BI", "SQL Server", "Python"],
        "category": "Finance",
        "table_name": "financial_reports"
    },
    {
        "project_id": "PROJ004",
        "project_name": "Q3 Marketing Campaign",
        "quarter": "Q3",
        "fiscal_year": "FY24",
        "start_date": "2023-10-01", 
        "end_date": "2023-12-31",
        "status": "Completed",
        "budget": 250000,
        "team_size": 7,
        "description": "Digital marketing campaign execution for Q3 FY24 targeting customer acquisition",
        "technologies": ["Google Ads", "Facebook Ads", "Analytics", "CRM"],
        "category": "Marketing",
        "table_name": "marketing_campaigns"
    }
]

def test_connection():
    """Test Elasticsearch connection"""
    try:
        response = requests.get(f"{ES_URL}/")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Connected to Elasticsearch: {data['cluster_name']}")
            return True
        else:
            print(f"❌ Connection failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Connection error: {str(e)}")
        return False

def create_index():
    """Create index if it doesn't exist"""
    try:
        # Check if index exists
        response = requests.head(f"{ES_URL}/{ES_INDEX}")
        if response.status_code == 200:
            print(f"✅ Index '{ES_INDEX}' already exists")
            return True
        
        # Create index
        index_settings = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0
            }
        }
        
        response = requests.put(
            f"{ES_URL}/{ES_INDEX}",
            headers={"Content-Type": "application/json"},
            data=json.dumps(index_settings)
        )
        
        if response.status_code in [200, 201]:
            print(f"✅ Created index: {ES_INDEX}")
            return True
        else:
            print(f"❌ Failed to create index: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error creating index: {str(e)}")
        return False

def ingest_documents():
    """Ingest sample documents"""
    success_count = 0
    
    for doc in sample_data:
        try:
            # Add metadata
            doc["ingestion_timestamp"] = datetime.now().isoformat()
            doc["document_id"] = f"doc_{doc['project_id']}"
            
            # Index document
            response = requests.put(
                f"{ES_URL}/{ES_INDEX}/_doc/{doc['project_id']}",
                headers={"Content-Type": "application/json"},
                data=json.dumps(doc)
            )
            
            if response.status_code in [200, 201]:
                result = response.json()
                success_count += 1
                print(f"✅ Indexed: {doc['project_name']} ({doc['project_id']}) - {result.get('result', 'unknown')}")
            else:
                print(f"❌ Failed to index {doc['project_id']}: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Error indexing {doc['project_id']}: {str(e)}")
    
    return success_count

def refresh_index():
    """Refresh index to make documents searchable"""
    try:
        response = requests.post(f"{ES_URL}/{ES_INDEX}/_refresh")
        if response.status_code == 200:
            print("✅ Index refreshed - documents are now searchable")
            return True
        else:
            print(f"⚠️ Failed to refresh index: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error refreshing index: {str(e)}")
        return False

def verify_data():
    """Verify ingested data"""
    try:
        # Count documents
        response = requests.get(f"{ES_URL}/{ES_INDEX}/_count")
        if response.status_code == 200:
            count = response.json()["count"]
            print(f"📊 Total documents in index: {count}")
            
            # Get sample document
            search_query = {"query": {"match_all": {}}, "size": 1}
            response = requests.get(
                f"{ES_URL}/{ES_INDEX}/_search",
                headers={"Content-Type": "application/json"},
                data=json.dumps(search_query)
            )
            
            if response.status_code == 200:
                results = response.json()
                if results["hits"]["hits"]:
                    sample = results["hits"]["hits"][0]["_source"]
                    print(f"📄 Sample document: {sample.get('project_name', 'Unknown')}")
            
            return True
        else:
            print(f"❌ Failed to verify data: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error verifying data: {str(e)}")
        return False

def main():
    print("🚀 Starting simple data ingestion for DataAssist POC...")
    
    # Test connection
    if not test_connection():
        print("\n💥 Cannot connect to Elasticsearch. Make sure it's running on localhost:9200")
        return False
    
    # Create index
    if not create_index():
        print("\n💥 Failed to create index")
        return False
    
    # Ingest documents
    success_count = ingest_documents()
    print(f"\n📈 Successfully indexed {success_count}/{len(sample_data)} documents")
    
    # Refresh index
    refresh_index()
    
    # Verify data
    if verify_data():
        print("\n🎉 Data ingestion completed successfully!")
        print("📝 Your DataAssist application now has sample data to work with.")
        print("🔍 Try asking questions like:")
        print("   - 'List Project Details for Q3'")
        print("   - 'Show me Q3 FY24 projects'")  
        print("   - 'What projects were completed in Q3?'")
        return True
    else:
        print("\n⚠️ Data ingestion completed but verification failed")
        return False

if __name__ == "__main__":
    main()