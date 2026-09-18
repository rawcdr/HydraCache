import os
import logging
import pdfplumber
from typing import List
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class PDFParser:
    def __init__(self, file_path: str, document_id: str = None):
        self.file_path = file_path
        self.document_id = document_id or os.path.basename(file_path).split('.')[0]

    def parse(self) -> List[Document]:
        """
        Extract tables and narrative text from the PDF using pdfplumber.
        Yields Document objects classified as 'table' or 'text'.
        """
        logger.info(f"Parsing PDF from {self.file_path} for Tables and Text...")
        docs = []
        
        with pdfplumber.open(self.file_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                # 1. Extract tables
                tables = page.find_tables()
                table_bboxes = []
                for table in tables:
                    table_bboxes.append(table.bbox)
                    extracted_table = table.extract()
                    if extracted_table:
                        # Convert to Markdown table manually
                        md_rows = []
                        for row_idx, row in enumerate(extracted_table):
                            clean_row = [str(cell).replace('\n', ' ') if cell else "" for cell in row]
                            md_rows.append("| " + " | ".join(clean_row) + " |")
                            if row_idx == 0:
                                md_rows.append("|" + "|".join(["---"] * len(clean_row)) + "|")
                        md_table = "\n".join(md_rows)
                        
                        doc = Document(
                            page_content=md_table,
                            metadata={
                                "document_id": self.document_id,
                                "source_file": self.file_path,
                                "page_number": page_number,
                                "chunk_type": "table",
                            }
                        )
                        docs.append(doc)
                
                # 2. Extract remaining text
                # We filter out any characters that fall inside table bounding boxes
                def not_within_bboxes(obj):
                    def obj_in_bbox(bbox):
                        v_match = (bbox[1] <= obj["top"] <= bbox[3]) or (bbox[1] <= obj["bottom"] <= bbox[3])
                        h_match = (bbox[0] <= obj["x0"] <= bbox[2]) or (bbox[0] <= obj["x1"] <= bbox[2])
                        return v_match and h_match
                    return not any(obj_in_bbox(bbox) for bbox in table_bboxes)

                text = page.filter(not_within_bboxes).extract_text()
                if text and text.strip():
                    doc = Document(
                        page_content=text.strip(),
                        metadata={
                            "document_id": self.document_id,
                            "source_file": self.file_path,
                            "page_number": page_number,
                            "chunk_type": "text",
                        }
                    )
                    docs.append(doc)
                    
        logger.info(f"Successfully extracted {len(docs)} logical elements from PDF.")
        return docs
