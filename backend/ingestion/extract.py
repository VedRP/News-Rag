import os
from dataclasses import dataclass, field
from typing import List, Tuple
import pymupdf

@dataclass
class ExtractedBlock:
    """A spatial text block from a PDF page."""
    text: str
    bbox: Tuple[float, float, float, float]
    block_no: int
    block_type: int  # 0: text, 1: image

@dataclass
class ExtractedPage:
    """Extracted content from a single PDF page."""
    page_number: int  # 1-indexed
    blocks: List[ExtractedBlock] = field(default_factory=list)
    raw_text: str = ""
    width: float = 0.0
    height: float = 0.0

@dataclass
class ExtractedDocument:
    """Complete extraction result for a PDF document."""
    source_filename: str
    file_path: str
    total_pages: int
    pages: List[ExtractedPage] = field(default_factory=list)

def extract_pdf(pdf_path: str) -> ExtractedDocument:
    """
    Extracts raw text and spatial blocks from a PDF file using PyMuPDF.
    
    Args:
        pdf_path: Absolute or relative path to the PDF file.
        
    Returns:
        ExtractedDocument containing page-by-page blocks and metadata.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    doc = pymupdf.open(pdf_path)
    filename = os.path.basename(pdf_path)
    pages: List[ExtractedPage] = []
    
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        rect = page.rect
        # Extract blocks with spatial reading order
        raw_blocks = page.get_text("blocks", sort=True)
        blocks: List[ExtractedBlock] = []
        
        for b in raw_blocks:
            # PyMuPDF block format: (x0, y0, x1, y1, text, block_no, block_type)
            x0, y0, x1, y1, text, block_no, block_type = b[:7]
            # Strip excessive null / control characters if present
            cleaned_block_text = text.replace("\x00", "").strip()
            if cleaned_block_text:
                blocks.append(
                    ExtractedBlock(
                        text=cleaned_block_text,
                        bbox=(x0, y0, x1, y1),
                        block_no=block_no,
                        block_type=block_type,
                    )
                )
        
        raw_page_text = page.get_text("text").strip()
        pages.append(
            ExtractedPage(
                page_number=page_idx + 1,
                blocks=blocks,
                raw_text=raw_page_text,
                width=rect.width,
                height=rect.height,
            )
        )
    
    doc.close()
    return ExtractedDocument(
        source_filename=filename,
        file_path=pdf_path,
        total_pages=len(pages),
        pages=pages,
    )
