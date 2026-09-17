import logging
import sys
import os

# Ensure the root path is appended for absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.ingestion.parser import PDFParser
from src.ingestion.chunker import TextChunker
from src.indexing.dense import DenseIndexer
from src.indexing.sparse import SparseIndexer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("==================================================")
    logger.info("Starting Phase 1 Indexing Pipeline")
    logger.info("==================================================")

    # 1. Parsing
    parser = PDFParser(Config.RAW_DATA_PATH)
    docs = parser.parse()

    # 2. Chunking
    chunker = TextChunker()
    chunks = chunker.chunk_documents(docs)

    # 3. Dense Indexing (Qdrant + FastEmbed)
    dense_indexer = DenseIndexer()
    dense_indexer.index(chunks)

    # 4. Sparse Indexing (BM25)
    sparse_indexer = SparseIndexer()
    sparse_indexer.index(chunks)

    logger.info("==================================================")
    logger.info("Phase 1 Indexing Pipeline completed successfully! 🎉")
    logger.info("==================================================")

if __name__ == "__main__":
    main()
