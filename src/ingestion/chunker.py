import logging
import re
import uuid
from typing import List
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class TextChunker:
    def __init__(self, window_size: int = 3, overlap: int = 1):
        self.window_size = window_size
        self.overlap = overlap

    def split_into_sentences(self, text: str) -> List[str]:
        """Regex-based sentence splitter."""
        # Split on . ! ? followed by space or newline
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Creates Sentence-Window child chunks linked to Parent chunks.
        Tables are maintained as single parent=child chunks.
        """
        logger.info(f"Chunking {len(documents)} documents using Sentence-Window strategy...")
        child_chunks = []
        
        for doc in documents:
            parent_id = str(uuid.uuid4())
            chunk_type = doc.metadata.get("chunk_type", "unknown")
            
            if chunk_type == "table":
                # Table is its own child and parent
                child_doc = Document(
                    page_content=doc.page_content,
                    metadata={
                        **doc.metadata,
                        "chunk_id": str(uuid.uuid4()),
                        "parent_id": parent_id,
                        "parent_text": doc.page_content
                    }
                )
                child_chunks.append(child_doc)
                
            elif chunk_type == "text":
                sentences = self.split_into_sentences(doc.page_content)
                if not sentences:
                    continue
                    
                parent_text = doc.page_content
                
                # Sliding window
                step = max(1, self.window_size - self.overlap)
                for i in range(0, len(sentences), step):
                    window = sentences[i:i + self.window_size]
                    if not window:
                        continue
                        
                    child_text = " ".join(window)
                    child_doc = Document(
                        page_content=child_text,
                        metadata={
                            **doc.metadata,
                            "chunk_id": str(uuid.uuid4()),
                            "parent_id": parent_id,
                            "parent_text": parent_text
                        }
                    )
                    child_chunks.append(child_doc)
                    
        logger.info(f"Generated {len(child_chunks)} child chunks with parent relationships.")
        return child_chunks
