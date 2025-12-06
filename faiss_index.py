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


# load chunked data that was created from embed_index.py
with open("data/chunked_data.pkl", "rb") as f:
    chunked_data = pickle.load(f)

# extract the embeddings and texts from the chunked data
embeddings = np.array([chunk.get("embedding") for chunk in chunked_data])
texts = [c["text"] for c in chunked_data]

# build FAISS index
dimension = embeddings.shape[1] # dimension of the embeddings
index = faiss.IndexFlatL2(dimension) # create a flat L2 index
index.add(embeddings) # add the embeddings to the index

# save the index to disk
faiss.write_index(index, "news_chunked.index")
with open("chunked_texts.pkl", "wb") as f:
    pickle.dump(texts, f)

# query test
model = SentenceTransformer("all-MiniLM-L6-v2")
query = "global markets reaction to interest rate hike"
query_embedding = model.encode([query]).astype('float32')
k = 5 # number of nearest neighbor to return
distances, indices = index.search(query_embedding, k)

# print the results
print(f"Nearest neighbors for query: {query}")
for dist, idx in zip(distances[0], indices[0]):
    print(f"Distance: {dist:.4f}, Text: {texts[idx][:200]}")