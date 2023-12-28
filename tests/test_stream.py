"""Test server search_results and stream_answer endpoints."""
import requests


# not a normal test. You must run the server first and then run this file.
if __name__ == "__main__":
    server = "http://127.0.0.1:8000"
    # server = "https://scripturecentralqa.org/api"
    url = f"{server}/search_results?q=who+is+oliver+cowdery"
    response = requests.get(
        url,
        timeout=30,
        headers={"accept": "application/json"},
    )
    print(response.headers)
    results = response.json()
    print(results["key"])
    print(len(results["results"]))

    url = f"{server}/stream_answer?key={results['key']}"
    response = requests.get(
        url,
        stream=True,
        timeout=5,
    )
    print(response.headers)

    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            print(str(chunk, encoding="utf-8"), end="")
