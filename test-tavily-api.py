import requests
import os
from dotenv import load_dotenv

load_dotenv()

url = "https://api.tavily.com/search"

payload = {
    "api_key": os.getenv("TAVILY_API_KEY"),
    "query": "What is RAG?",
    "max_results": 3
}

response = requests.post(url, json=payload)
print(response.json())