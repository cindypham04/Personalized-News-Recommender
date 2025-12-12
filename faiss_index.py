import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

'''
How FAISS works: 
1. embed_index.py: chunk each article into smaller pieces → embed → get a vector representation.
2. faiss_index.py: add each vector to FAISS index (with a unique integer ID), and store a separate mapping from ID 
→ metadata (article title, URL, chunk index, etc.).
3. On user query:
- Embed the query text → vector.
- Search FAISS for nearest neighbors → get top-k vectors (IDs + distances).
- Translate IDs → metadata → fetch the corresponding chunks/articles.
- Use those chunks (or whole articles) as output — ranking, summarization, etc.
'''

# ------ Build the FAISS index ------

# load chunked data that was created from embed_index.py
with open("data/chunked_data.pkl", "rb") as f:
    chunked_data = pickle.load(f)

# extract the embeddings and texts from the chunked data
embeddings = np.array([chunk.get("embedding") for chunk in chunked_data], dtype=np.float32)
# Ensure the array is contiguous in memory
embeddings = np.ascontiguousarray(embeddings)
texts = [c["text"] for c in chunked_data]

# metadata 
metadata = [
    {
        "article_idx": chunk.get("article_idx"),
        "chunk_idx": chunk.get("chunk_idx"),
        "text": chunk.get("text"),
        "title": chunk.get("title"),
        "url": chunk.get("url"),
        "published_at": chunk.get("published_at"),
    }
    for chunk in chunked_data
]

# build FAISS index
dimension = embeddings.shape[1] # dimension of the embeddings

# normalize for cosine similarity
faiss.normalize_L2(embeddings)

# inner product = cosine for normalized vectors 
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# save the index to disk
faiss.write_index(index, "news_chunked.index")
with open("chunked_texts.pkl", "wb") as f:
    pickle.dump(metadata, f)

print(f"Indexed {index.ntotal} chunks")

# ------ Query the FAISS index ------
# query test
model = SentenceTransformer("all-MiniLM-L6-v2")

# Note: If you want to reload the index from disk, uncomment the lines below:
# index = faiss.read_index("news_chunked.index")
# with open("chunked_texts.pkl", "rb") as f: 
#     metadata = pickle.load(f)
# Otherwise, we'll use the index and metadata already in memory

def dense_search(query, top_k=5):

    # embed the query
    query_vec = model.encode([query]).astype('float32')
    # Ensure the array is contiguous in memory
    query_vec = np.ascontiguousarray(query_vec)

    # normalize for cosine similarity
    faiss.normalize_L2(query_vec)

    scores, indices = index.search(query_vec, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "score": float(score), 
            **metadata[idx]
        })
    
    return results

results = dense_search("donald trump on tariffs", top_k=5)

for r in results:
    print(f"{r['score']:.3f} | {r['title']}")
