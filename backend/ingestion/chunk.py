import re
from dataclasses import dataclass
from typing import List, Optional
from .extract import ExtractedBlock, ExtractedPage

@dataclass
class ArticleChunk:
    """A coherent article chunk extracted from a newspaper page."""
    title: str
    text: str
    page: int
    word_count: int
    section: Optional[str] = None

# Patterns that signal new articles or bylines
DATELINE_PATTERNS = [
    re.compile(r"^(?:Press Trust of India|PTI|ANI|Reuters|Special Correspondent|Staff Reporter)\b", re.IGNORECASE),
    re.compile(r"^[A-Z\s]{3,20}\s*,\s*(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\.?\s*\d{1,2}", re.IGNORECASE),
    re.compile(r"^(?:NEW DELHI|MUMBAI|BENGALURU|KOLKATA|CHENNAI|HYDERABAD|VADODARA|PUNE|AHMEDABAD)\s*$", re.IGNORECASE),
]

SECTION_HEADERS = {
    "IN BRIEF", "STATES", "CITY", "NATION", "WORLD", "BUSINESS", "SPORT", "EDITORIAL", "OPINION"
}

def is_likely_headline(text: str) -> bool:
    """Detects whether a short text block is likely an article headline."""
    stripped = text.strip()
    lines = [l.strip() for l in stripped.split("\n") if l.strip()]
    if not lines:
        return False
        
    first_line = lines[0]
    total_len = len(stripped)
    
    # Very long blocks are body text, not headlines
    if total_len > 220:
        return False
        
    # Check if section header
    if stripped.upper() in SECTION_HEADERS or any(s in stripped.upper() for s in ["» PAGE", "IN BRIEF"]):
        return True

    # If it ends with period, likely a regular sentence
    if stripped.endswith(".") and not any(title_abbr in stripped for title_abbr in ["Mr.", "Dr.", "Govt."]):
        return False

    # Headline characteristics: shorter, title casing or significant capital letters
    words = stripped.split()
    if 2 <= len(words) <= 25:
        # Check if first character is uppercase
        if first_line[0].isupper():
            return True

    return False

def split_large_article(title: str, text: str, page: int, max_words: int = 500) -> List[ArticleChunk]:
    """Splits an unusually long article into paragraphs while preserving title context."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[ArticleChunk] = []
    
    current_paragraphs: List[str] = []
    current_word_count = 0
    
    for p in paragraphs:
        p_words = len(p.split())
        if current_word_count + p_words > max_words and current_paragraphs:
            chunk_body = "\n\n".join(current_paragraphs)
            full_text = f"{title}\n\n{chunk_body}" if title and not chunk_body.startswith(title) else chunk_body
            chunks.append(
                ArticleChunk(
                    title=title,
                    text=full_text,
                    page=page,
                    word_count=len(full_text.split()),
                )
            )
            current_paragraphs = [p]
            current_word_count = p_words
        else:
            current_paragraphs.append(p)
            current_word_count += p_words
            
    if current_paragraphs:
        chunk_body = "\n\n".join(current_paragraphs)
        full_text = f"{title}\n\n{chunk_body}" if title and not chunk_body.startswith(title) else chunk_body
        chunks.append(
            ArticleChunk(
                title=title,
                text=full_text,
                page=page,
                word_count=len(full_text.split()),
            )
        )
        
    return chunks

def chunk_newspaper_page(cleaned_blocks: List[ExtractedBlock], page_number: int) -> List[ArticleChunk]:
    """
    Groups cleaned spatial blocks on a page into coherent article chunks.
    """
    if not cleaned_blocks:
        return []
        
    stories: List[List[ExtractedBlock]] = []
    current_story_blocks: List[ExtractedBlock] = []
    
    for block in cleaned_blocks:
        txt = block.text.strip()
        
        # New article detection: if block is a headline and we already accumulated enough text
        if is_likely_headline(txt) and current_story_blocks:
            # Check accumulated words in current story
            total_words = sum(len(b.text.split()) for b in current_story_blocks)
            if total_words >= 40:
                # Commit previous story and start new one
                stories.append(current_story_blocks)
                current_story_blocks = [block]
                continue
                
        current_story_blocks.append(block)
        
    if current_story_blocks:
        stories.append(current_story_blocks)
        
    final_chunks: List[ArticleChunk] = []
    
    for story_blocks in stories:
        combined_text = "\n\n".join(b.text for b in story_blocks if b.text.strip()).strip()
        words = combined_text.split()
        if len(words) < 25:
            # Skip tiny orphaned snippets (less than 25 words)
            continue
            
        # Determine title from first block or headline candidate
        first_txt = story_blocks[0].text.strip()
        title = first_txt.split("\n")[0] if len(first_txt.split("\n")[0]) < 150 else first_txt[:100]
        
        if len(words) > 600:
            sub_chunks = split_large_article(title, combined_text, page_number, max_words=500)
            final_chunks.extend(sub_chunks)
        else:
            final_chunks.append(
                ArticleChunk(
                    title=title,
                    text=combined_text,
                    page=page_number,
                    word_count=len(words),
                )
            )
            
    return final_chunks
