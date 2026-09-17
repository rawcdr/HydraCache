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
    parent_id: Optional[str] = None
    retrieval_source: str = "unknown"
    provenance: Optional[List[str]] = None
    rrf_score: float = 0.0
