import os
import json
import requests
from newspaper import Article
from datetime import datetime

API_KEY = "6e25498b6e074aef984268518d05dcdc"
NEWSAPI_URL = "https://newsapi.org/v2/top-headlines"

def fetch_headlines(country="us", page_size=50):
    # params for the API request 
    params = {
        "apiKey": API_KEY,
        "country": country,
        "pageSize": page_size, # maximum number of articles to return
    }

    # make the API request
    resp = requests.get(NEWSAPI_URL, params=params)

    # raise an exception if request fails
    resp.raise_for_status()

    # save the response as JSON
    data = resp.json() 
    return data.get("articles", [])

def extract_full_text(url):
    try: 
        # initialize the newspaper article object
        art = Article(url)

        # download the article
        art.download()

        # parse the article
        art.parse()

        # return the full text of the article
        return art.text

    except Exception as e:
        print(f"Error extracting full text from {url}: {e}")
        return None

def normalize_and_save(articles, output_path="data/raw_news.json"):
    # store the cleaned articles
    cleaned = [] 

    for a in articles:
        url = a.get("url")

        # skip if the article has no url
        if not url: 
            continue

        # extract the full text of the article
        body = extract_full_text(url) 

        # skip if the article is too short
        if not body or len(body.split()) < 100: 
            continue

        cleaned.append({
            "title":a.get("title"),
            "url":url,
            "published_at": a.get("publishedAt"),
            "source": a.get("source", {}).get("name"),
            "body": body,
        })

        # write to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, ensure_ascii=False, indent=4) 

if __name__ == "__main__":
    head = fetch_headlines()
    normalize_and_save(head)
    print("Fetched and saved", len(head), "headlines")