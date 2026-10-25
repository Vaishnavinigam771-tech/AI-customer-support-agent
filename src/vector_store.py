from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from .config import (
    EMBEDDING_MODEL,
    CHROMA_DIR,
    COLLECTION_NAME
)


def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


def create_vector_store(documents):

    embeddings = get_embeddings()

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR
    )

    return vector_store


def load_vector_store():

    embeddings = get_embeddings()

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )