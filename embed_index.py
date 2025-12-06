import json
from sentence_transformers import SentenceTransformer
import nltk 

# download the punkt tokenizer
nltk.download('punkt')

from nltk.tokenize import word_tokenize

# load the articles that are stored under data/raw_news.json
with open("data/raw_news.json", "r") as f:
    articles = json.load(f)

# initialize the sentence transformer embdding model
model = SentenceTransformer("all-MiniLM-L6-v2")