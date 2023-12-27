"""import request module."""
import requests


url = "http://127.0.0.1:8000/stream_answer?key=1LhauY9cIy"
response = requests.get(
    url,
    stream=True,
    timeout=5,
    # headers={"accept": "application/json"},
)

for chunk in response.iter_content(chunk_size=1024):
    if chunk:
        print(str(chunk, encoding="utf-8"), end="")
