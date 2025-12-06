import json
from sentence_transformers import SentenceTransformer
import nltk 
import pickle

# download the punkt tokenizer
nltk.download('punkt_tab')

from nltk.tokenize import sent_tokenize

# load the articles that are stored under data/raw_news.json
with open("data/raw_news.json", "r") as f:
    articles = json.load(f)

# initialize the sentence transformer embdding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_text(text, max_sentences=5):
    # split a block of text into sentences 
    sentences = sent_tokenize(text)
    chunks = []

    # chunk the text into chunks of 5 sentences
    for i in range(0, len(sentences), max_sentences):

        # join the sentences into a single chunk
        chunk = " ".join(sentences[i:i+max_sentences]) 

        chunks.append(chunk)

    return chunks

chunked_data = [] # list of dicts: {article_id, chunk_id, chunk_text, embedding}

# embed all articles by chunk
for idx, article in enumerate(articles):
    body = article.get("body")

    # skip if the article has no body
    if not body:
        continue

    chunks = chunk_text(body, max_sentences=5)
    embeddings = model.encode(chunks, batch_size=8) # this returns a list of embeddings

    for chunk_idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chunked_data.append({
            "article_idx": idx,
            "chunk_idx": chunk_idx, 
            "text": chunk,
            "embedding": embedding.tolist(),
            "title": article.get("title"),
            "url": article.get("url"),
            "published_at": article.get("published_at"),
        })

print(f"Created {len(chunked_data)} chunks from {len(articles)} articles")

# save the chunked data to a pickle file
with open("data/chunked_data.pkl", "wb") as f:
    pickle.dump(chunked_data, f)
