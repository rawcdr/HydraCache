import logging
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.config import Config

logger = logging.getLogger(__name__)

class TextChunker:
    def __init__(self, chunk_size: int = Config.CHUNK_SIZE, chunk_overlap: int = Config.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Takes raw page documents and splits them into smaller semantic chunks.
        """
        logger.info(f"Splitting {len(documents)} documents (chunk_size={self.chunk_size}, overlap={self.chunk_overlap})...")
        chunks = self.splitter.split_documents(documents)
        logger.info(f"Generated {len(chunks)} chunks.")
        
        # Optionally, inject chunk IDs or structural metadata here
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = f"chunk_{i}"
            
        return chunks
