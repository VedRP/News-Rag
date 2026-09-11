import os
from dataclasses import dataclass, field
from typing import List, Tuple
import pymupdf

# Above this fraction of non-text characters in the extracted text, treat the PDF as
# having broken font encoding rather than genuine content (see GarbledPDFTextError).
# Real newspaper PDFs we've tested land under 1%; a broken one we hit was ~89%.
GARBLED_TEXT_RATIO_THRESHOLD = 0.3
MIN_CHARS_FOR_GARBLE_CHECK = 200


class GarbledPDFTextError(Exception):
    """
    Raised when a PDF's text layer extracts as mostly non-text characters.

    This happens when a PDF embeds subsetted CID fonts (Type0, Identity-H encoding)
    without a usable ToUnicode CMap: the PDF renders correctly visually, but there is
    no way to recover which Unicode character each glyph represents, so extracted
    "text" is essentially scrambled. This is NOT the same as a scanned/image-only PDF
    (which task.md separately calls out as needing OCR) -- there IS a text layer here,
    it's just unrecoverable without the font's original CMap. There is no reliable
    automatic fix; the PDF needs to be re-exported/re-sourced with proper text encoding.
    """


def _non_text_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    non_text = sum(
        1 for ch in text
        if not (ch.isalnum() or ch.isspace() or ch in ".,;:!?-‘’“”–—'\"()[]/%&$@#")
    )
    return non_text / len(text)


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

    combined_text = "".join(p.raw_text for p in pages)
    if len(combined_text) >= MIN_CHARS_FOR_GARBLE_CHECK:
        ratio = _non_text_char_ratio(combined_text)
        if ratio > GARBLED_TEXT_RATIO_THRESHOLD:
            raise GarbledPDFTextError(
                f"{filename}: extracted text is {ratio:.0%} non-text characters "
                f"(threshold {GARBLED_TEXT_RATIO_THRESHOLD:.0%}). This PDF likely has "
                "subsetted fonts without a usable ToUnicode CMap, so its text layer "
                "cannot be reliably recovered. Try a different export/source of this PDF."
            )

    return ExtractedDocument(
        source_filename=filename,
        file_path=pdf_path,
        total_pages=len(pages),
        pages=pages,
    )
