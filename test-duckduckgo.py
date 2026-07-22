import requests

url = "https://api.duckduckgo.com/"
params = {
    "q": "What is RAG?",
    "format": "json",
    "no_html": 1
}

response = requests.get(url, params=params)
print(response.json())