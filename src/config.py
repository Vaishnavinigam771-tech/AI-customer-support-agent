import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

LLM_MODEL = "openai/gpt-oss-20b"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHROMA_DIR = "./chroma_db"

COLLECTION_NAME = "customer_support"

TOP_K = 4

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100