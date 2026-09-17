import pytest
from langchain_core.documents import Document
from src.ingestion.chunker import TextChunker

def test_sentence_window_chunker():
    text = "Sentence one. Sentence two! Sentence three? Sentence four."
    doc = Document(page_content=text, metadata={"chunk_type": "text"})
    
    chunker = TextChunker(window_size=2, overlap=1)
    chunks = chunker.chunk_documents([doc])
    
    assert len(chunks) == 4
    assert chunks[0].page_content == "Sentence one. Sentence two!"
    assert chunks[1].page_content == "Sentence two! Sentence three?"
    assert chunks[2].page_content == "Sentence three? Sentence four."
    assert chunks[3].page_content == "Sentence four."
    
    for chunk in chunks:
        assert "chunk_id" in chunk.metadata
        assert "parent_id" in chunk.metadata
        assert chunk.metadata["parent_text"] == text

def test_table_chunker():
    doc = Document(page_content="| A | B |\n|---|---|", metadata={"chunk_type": "table"})
    chunker = TextChunker()
    chunks = chunker.chunk_documents([doc])
    
    assert len(chunks) == 1
    assert chunks[0].page_content == "| A | B |\n|---|---|"
    assert "parent_id" in chunks[0].metadata
