import pytest
import os
from langchain_core.documents import Document
from src.indexing.sparse import SparseIndexer

def test_sparse_indexer(tmp_path):
    index_path = os.path.join(tmp_path, "bm25_test.pkl")
    indexer = SparseIndexer(save_path=index_path)
    
    docs = [
        Document(page_content="Apple is a global tech company.", metadata={"chunk_id": "c1"}),
        Document(page_content="The iPhone is their most popular product.", metadata={"chunk_id": "c2"})
    ]
    
    indexer.index(docs)
    
    assert os.path.exists(index_path)
    assert indexer.bm25 is not None
    assert indexer.bm25.corpus_size == 2
