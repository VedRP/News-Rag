import re
from typing import List
from .extract import ExtractedBlock

# Regex patterns for newspaper noise
NOISE_LINE_PATTERNS = [
    r"^[JM]\s+ND-NDE\s+CM\s+YK.*$",
    r"^CM\s+YK.*$",
    r"^[JM]\d+$",
    r"^\d{5,8}$",  # Margin barcodes/tracking numbers (e.g. 365565)
    r"^Regd\.\s*DL\(ND\)-[\w/-]+.*$",
    r"^RNI\s*No\.\s*[\w/-]+.*$",
    r"^Vol\.\s*\d+\s+No\.\s*\d+.*$",
    r"^\d+\s+Pages\s+₹\s*[\d.]+.*$",
    r"^www\.\w+\.com.*$",
    r"^To subscribe,.*$",
    r"^give a missed call at.*$",
    r"^or scan QR code.*$",
    r"^Printed at.*$",
    r"^»\s*\w+.*$",
]

COMPILED_NOISE = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in NOISE_LINE_PATTERNS]

def is_boilerplate_block(text: str) -> bool:
    """Checks if a whole block is boilerplate publishing noise."""
    stripped = text.strip()
    if not stripped:
        return True
    
    # Check if single line or very short noise
    for pattern in COMPILED_NOISE:
        if pattern.match(stripped):
            return True
            
    # Margin barcode or printer alignment artifact
    if re.fullmatch(r"\d{5,8}", stripped):
        return True
    if re.fullmatch(r"[A-Z]{1,3}\s+[A-Z0-9-]+\s+[A-Z]{2}\s+[A-Z]{2}", stripped):
        return True
        
    return False

def clean_text(text: str) -> str:
    """
    Cleans raw text:
    - Removes known boilerplate lines
    - De-hyphenates column wrap words (e.g. "dedicat-\ned" -> "dedicated")
    - Normalizes single newlines inside sentences to spaces
    - Collapses excessive blank lines
    """
    if not text:
        return ""
    
    # 1. Remove noise lines
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if any(p.match(stripped) for p in COMPILED_NOISE):
            continue
        cleaned_lines.append(line)
    
    content = "\n".join(cleaned_lines)
    
    # 2. De-hyphenate broken words across newlines (newspaper column break)
    content = re.sub(r"(\b[A-Za-z]+)-\n([A-Za-z]+\b)", r"\1\2", content)
    
    # 3. Fix soft hyphen characters (\xad)
    content = content.replace("\xad", "")
    
    # 4. Join lines that are part of the same sentence
    # Preserve double newlines (paragraphs), convert single newlines to spaces
    paragraphs = content.split("\n\n")
    normalized_paragraphs = []
    for p in paragraphs:
        # Replace remaining single newlines with space
        joined = re.sub(r"(?<!\n)\n(?!\n)", " ", p).strip()
        joined = re.sub(r"\s{2,}", " ", joined)
        if joined:
            normalized_paragraphs.append(joined)
            
    return "\n\n".join(normalized_paragraphs)

def clean_page_blocks(blocks: List[ExtractedBlock]) -> List[ExtractedBlock]:
    """
    Filters and cleans spatial blocks on a page:
    - Drops boilerplate header/margin/barcode blocks
    - Reassembles dropped capital letters (e.g. 'P' + 'rime Minister' -> 'Prime Minister')
    - Cleans internal text
    """
    filtered: List[ExtractedBlock] = []
    i = 0
    
    while i < len(blocks):
        b = blocks[i]
        txt = b.text.strip()
        
        # Check for margin barcode or known noise
        if is_boilerplate_block(txt):
            i += 1
            continue
        
        # Check for drop cap: single uppercase letter block followed by lowercase word continuation
        if len(txt) == 1 and txt.isupper() and (i + 1 < len(blocks)):
            next_b = blocks[i + 1]
            next_txt = next_b.text.strip()
            # If next block starts with lowercase letter or word fragment
            if next_txt and next_txt[0].islower():
                merged_text = txt + next_txt
                filtered.append(
                    ExtractedBlock(
                        text=clean_text(merged_text),
                        bbox=next_b.bbox,
                        block_no=next_b.block_no,
                        block_type=next_b.block_type,
                    )
                )
                i += 2
                continue
                
        cleaned_txt = clean_text(txt)
        if cleaned_txt:
            filtered.append(
                ExtractedBlock(
                    text=cleaned_txt,
                    bbox=b.bbox,
                    block_no=b.block_no,
                    block_type=b.block_type,
                )
            )
        i += 1
        
    return filtered
