import pytest
from langchain_core.documents import Document
from src.ingestion.chunker import TextChunker

def test_text_chunker():
    text = "This is a long document designed to test the chunking capabilities. " * 50
    doc = Document(page_content=text)
    
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk_documents([doc])
    
    assert len(chunks) > 1
    assert "chunk_id" in chunks[0].metadata
    assert chunks[0].metadata["chunk_id"] == "chunk_0"
