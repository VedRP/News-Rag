import os
from typing import List, Dict, Any, Optional
from .extract import extract_pdf
from .clean import clean_page_blocks
from .chunk import chunk_newspaper_page
from .metadata import extract_metadata, extract_date_from_text

def process_newspaper_pdf(
    pdf_path: str,
    max_pages: Optional[int] = None,
    language: str = "english",
) -> List[Dict[str, Any]]:
    """
    Executes the full Phase 1 ingestion pipeline:
    PDF -> Extract blocks -> Clean noise & de-hyphenate -> Chunk into articles -> Tag metadata
    
    Args:
        pdf_path: Path to the PDF file.
        max_pages: Optional limit on pages to process (for rapid testing).
        language: Primary language of the newspaper.
        
    Returns:
        List of chunk dictionaries with full metadata adhering to context.md Section 7.
    """
    doc = extract_pdf(pdf_path)
    all_chunks: List[Dict[str, Any]] = []
    
    # Try to extract the issue date from the first page header
    doc_date = "2026-09-09"
    if doc.pages:
        found_date = extract_date_from_text(doc.pages[0].raw_text)
        if found_date:
            doc_date = found_date
            
    pages_to_process = doc.pages[:max_pages] if max_pages else doc.pages
    
    for page in pages_to_process:
        cleaned_blocks = clean_page_blocks(page.blocks)
        page_chunks = chunk_newspaper_page(cleaned_blocks, page.page_number)
        
        for chk in page_chunks:
            chunk_dict = extract_metadata(
                chunk=chk,
                source_filename=doc.source_filename,
                default_date=doc_date,
                language=language,
            )
            all_chunks.append(chunk_dict)
            
    return all_chunks
