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
        logger.info(f"Building/Updating BM25 sparse index with {len(chunks)} new chunks...")
        
        all_chunks = chunks
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, 'rb') as f:
                    data = pickle.load(f)
                    if 'chunks' in data:
                        existing_chunks = data['chunks']
                        logger.info(f"Loaded {len(existing_chunks)} existing chunks from {self.save_path}")
                        # Filter out existing chunks that might have the same document_id if we wanted to replace them.
                        # For Phase 6, we'll just append for simplicity or assume different docs.
                        # Wait, we should probably check if chunks exist by chunk_id, but the simplest approach is just append.
                        # Wait, a safer approach is to deduplicate by chunk_id.
                        existing_chunk_ids = {c.metadata.get("chunk_id") for c in existing_chunks}
                        new_chunks = [c for c in chunks if c.metadata.get("chunk_id") not in existing_chunk_ids]
                        all_chunks = existing_chunks + new_chunks
            except Exception as e:
                logger.error(f"Failed to load existing BM25 index: {e}. Rebuilding from scratch.")

        self.chunks = all_chunks
        tokenized_corpus = [self.tokenize(chunk.page_content) for chunk in self.chunks]
        
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"BM25 index built successfully with {len(self.chunks)} total chunks.")
        self.save()

    def save(self):
        """Saves the BM25 object and the chunks for retrieval."""
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with open(self.save_path, 'wb') as f:
            pickle.dump({'bm25': self.bm25, 'chunks': self.chunks}, f)
        logger.info(f"Saved BM25 index to {self.save_path}")
