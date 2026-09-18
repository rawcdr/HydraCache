import time
import requests
import os
import sys

# Ensure root path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config

BASE_URL = "http://localhost:8000"
API_KEY = Config.API_KEY
HEADERS = {"X-API-Key": API_KEY}

def print_separator(title):
    print(f"\n{'='*50}\n{title}\n{'='*50}")

def test_health():
    print_separator("1. Testing GET /health")
    resp = requests.get(f"{BASE_URL}/health")
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.json()}")

def test_unauthenticated_query():
    print_separator("2. Testing Unauthenticated /query")
    resp = requests.post(f"{BASE_URL}/query", json={"query": "test"})
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.json()}")

def test_authenticated_query():
    print_separator("3. Testing Authenticated /query")
    resp = requests.post(
        f"{BASE_URL}/query", 
        json={"query": "What were Apple's net sales in 2025?"}, 
        headers=HEADERS
    )
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"Answer: {data['answer'][:100]}...")
        print(f"Cache Hit: {data['cache_hit']}")
        print(f"Latency: {data['latency']['total_latency']:.2f}ms")
    else:
        print(f"Error Response: {resp.json()}")

def test_repeated_query():
    print_separator("4. Testing Repeated /query (Should hit cache)")
    resp = requests.post(
        f"{BASE_URL}/query", 
        json={"query": "What were Apple's net sales in 2025?"}, 
        headers=HEADERS
    )
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"Cache Hit: {data['cache_hit']}")
        print(f"Latency: {data['latency']['total_latency']:.2f}ms")

def test_rate_limit():
    print_separator("5. Testing Rate Limits")
    print("Sending 12 quick requests...")
    for i in range(12):
        resp = requests.get(f"{BASE_URL}/health") # Health doesn't have rate limit
        resp = requests.get(f"{BASE_URL}/documents/invalid-id", headers=HEADERS)
        if resp.status_code == 429:
            print(f"Request {i+1}: Rate Limit Hit! Status: {resp.status_code}, Retry-After: {resp.headers.get('retry-after')}")
            break
        else:
            print(f"Request {i+1}: Status {resp.status_code}")

def test_document_endpoints():
    print_separator("6. Testing Document Upload and Polling")
    # Create a dummy PDF
    dummy_pdf_path = "data/raw/dummy.pdf"
    os.makedirs("data/raw", exist_ok=True)
    with open(dummy_pdf_path, "w") as f:
        f.write("dummy pdf") # Not a real PDF, but API only checks content-type header for now

    with open(dummy_pdf_path, "rb") as f:
        files = {"file": ("dummy.pdf", f, "application/pdf")}
        resp = requests.post(f"{BASE_URL}/documents", headers=HEADERS, files=files)
        
    print(f"Upload Status: {resp.status_code}")
    if resp.status_code == 200:
        job_id = resp.json()["job_id"]
        print(f"Job ID: {job_id}")
        
        print("\nPolling job status...")
        resp = requests.get(f"{BASE_URL}/documents/{job_id}", headers=HEADERS)
        print(f"Status check response: {resp.json()}")

if __name__ == "__main__":
    print("Ensure the FastAPI server is running via `uvicorn src.api.main:app --reload`")
    test_health()
    test_unauthenticated_query()
    test_authenticated_query()
    test_repeated_query()
    test_rate_limit()
    test_document_endpoints()
