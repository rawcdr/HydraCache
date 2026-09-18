from dataclasses import dataclass
from typing import Optional, List

@dataclass
class RetrievalResult:
    chunk_id: str
    text: str
    score: float
    source_file: str
    page_number: int
    chunk_type: str
    document_id: Optional[str] = None
    parent_id: Optional[str] = None
    retrieval_source: str = "unknown"
    provenance: Optional[List[str]] = None
    rrf_score: float = 0.0
    rerank_score: Optional[float] = None
