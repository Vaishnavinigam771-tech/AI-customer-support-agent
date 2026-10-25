from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CHUNK_SIZE, CHUNK_OVERLAP


def load_documents():

    documents = []

    folder = Path("data/faqs")

    for file_path in folder.glob("*.txt"):

        text = file_path.read_text(
            encoding="utf-8"
        )

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": file_path.name
                }
            )
        )

    return documents


def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    return splitter.split_documents(documents)