import logging
from typing import List
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class PDFParser:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def parse(self) -> List[Document]:
        """
        Extract text from the PDF using pdfplumber.
        Returns a list of LangChain Document objects.
        """
        logger.info(f"Parsing PDF from {self.file_path}...")
        loader = PDFPlumberLoader(self.file_path)
        docs = loader.load()
        logger.info(f"Successfully loaded {len(docs)} pages from PDF.")
        return docs
