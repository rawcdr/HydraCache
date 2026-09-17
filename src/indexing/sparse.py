import logging
import pickle
import os
from typing import List
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from src.config import Config

logger = logging.getLogger(__name__)

class SparseIndexer:
    def __init__(self, save_path: str = Config.BM25_INDEX_PATH):
        self.save_path = save_path
        self.bm25 = None
        self.chunks = None

    def tokenize(self, text: str) -> List[str]:
        # Simple whitespace tokenization; robust enough for Phase 1 baseline
        return text.lower().split()

    def index(self, chunks: List[Document]):
        """
        Creates a BM25 index from the documents and saves it to disk.
        """
        logger.info(f"Building BM25 sparse index for {len(chunks)} chunks...")
        self.chunks = chunks
        tokenized_corpus = [self.tokenize(chunk.page_content) for chunk in chunks]
        
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info("BM25 index built successfully.")
        self.save()

    def save(self):
        """Saves the BM25 object and the chunks for retrieval."""
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with open(self.save_path, 'wb') as f:
            pickle.dump({'bm25': self.bm25, 'chunks': self.chunks}, f)
        logger.info(f"Saved BM25 index to {self.save_path}")
