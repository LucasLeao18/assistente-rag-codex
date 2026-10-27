"""Configurações centrais do assistente RAG (lidas do ambiente / .env)."""
import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
COLLECTION = os.getenv("CHROMA_COLLECTION", "documentos")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "650"))        # tokens
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))  # tokens
TOP_K = int(os.getenv("TOP_K", "4"))
MIN_RELEVANCE = float(os.getenv("MIN_RELEVANCE", "0.30"))  # US06: limiar do "não encontrei"

NOT_FOUND_MESSAGE = "Não encontrei essa informação nos documentos enviados."
