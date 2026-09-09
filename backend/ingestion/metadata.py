import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from .chunk import ArticleChunk

# Mapping of cities to state/country for hierarchical location tagging
CITY_STATE_MAP = {
    "mumbai": "Maharashtra",
    "pune": "Maharashtra",
    "nagpur": "Maharashtra",
    "navi mumbai": "Maharashtra",
    "delhi": "Delhi",
    "new delhi": "Delhi",
    "noida": "Uttar Pradesh",
    "lucknow": "Uttar Pradesh",
    "vadodara": "Gujarat",
    "ahmedabad": "Gujarat",
    "surat": "Gujarat",
    "bengaluru": "Karnataka",
    "hubballi": "Karnataka",
    "chennai": "Tamil Nadu",
    "madurai": "Tamil Nadu",
    "coimbatore": "Tamil Nadu",
    "hyderabad": "Telangana",
    "kolkata": "West Bengal",
    "patna": "Bihar",
    "kochi": "Kerala",
    "thiruvananthapuram": "Kerala",
    "chandigarh": "Punjab",
    "mohali": "Punjab",
    "cuttack": "Odisha",
    "bhubaneswar": "Odisha",
    "jaipur": "Rajasthan",
}

INDIAN_STATES = [
    "maharashtra", "gujarat", "delhi", "karnataka", "tamil nadu",
    "telangana", "west bengal", "uttar pradesh", "bihar", "kerala",
    "punjab", "haryana", "rajasthan", "odisha", "andhra pradesh", "madhya pradesh"
]

TOPIC_KEYWORDS = {
    "cricket": ["cricket", "bcci", "ipl", "test match", "odi", "t20", "wicket", "batsman", "bowler", "kohli", "rohit"],
    "sports": ["sport", "football", "badminton", "olympics", "hockey", "tennis", "athlete", "championship", "tournament"],
    "politics": ["minister", "prime minister", "modi", "rahul gandhi", "congress", "bjp", "election", "parliament", "lok sabha", "mla", "mp", "opposition", "assembly"],
    "infrastructure": ["freight corridor", "highway", "railway", "expressway", "metro", "bridge", "inaugurate", "transport"],
    "business": ["business", "trade", "export", "import", "shares", "stock", "corporate", "industry", "revenue", "firm"],
    "finance": ["finance", "rbi", "bank", "rupee", "sensex", "nifty", "inflation", "tax", "gdp", "market", "economy"],
    "judiciary": ["court", "high court", "supreme court", "judge", "bench", "bail", "verdict", "ed", "cbi", "fir", "police"],
    "weather": ["weather", "rainfall", "rain", "monsoon", "flood", "temperature", "cyclone", "heatwave", "storm", "forecast"],
    "health": ["health", "hospital", "aiims", "doctor", "medical", "disease", "patient", "vaccine", "medicine"],
    "technology": ["technology", "ai", "software", "tech", "digital", "cyber", "internet", "startup", "data"],
    "science": ["science", "isro", "nasa", "space", "satellite", "research", "study", "scientists"],
    "education": ["education", "school", "university", "college", "student", "cbse", "ugc", "exam", "admission"],
}

DATE_PATTERNS = [
    re.compile(r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}", re.IGNORECASE),
    re.compile(r"\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}", re.IGNORECASE),
    re.compile(r"\d{4}-\d{2}-\d{2}"),
]

def extract_date_from_text(text: str) -> Optional[str]:
    """Finds publication date in text and formats as YYYY-MM-DD."""
    for pattern in DATE_PATTERNS:
        match = pattern.search(text)
        if match:
            date_str = match.group(0)
            for fmt in ("%B %d, %Y", "%d %B %Y", "%Y-%m-%d"):
                try:
                    dt = datetime.strptime(date_str, fmt)
                    return dt.strftime("%Y-%m-%d")
                except ValueError:
                    continue
    return None

def detect_locations(text: str) -> List[str]:
    """Extracts city and state names mentioned in article text."""
    lower_text = text.lower()
    locations: set[str] = set()
    
    for city, state in CITY_STATE_MAP.items():
        # Match whole word
        if re.search(r"\b" + re.escape(city) + r"\b", lower_text):
            locations.add(city.title())
            locations.add(state)
            
    for state in INDIAN_STATES:
        if re.search(r"\b" + re.escape(state) + r"\b", lower_text):
            locations.add(state.title())
            
    return sorted(list(locations))

def detect_topics(text: str) -> List[str]:
    """Extracts relevant topics based on keyword presence."""
    lower_text = text.lower()
    topics: set[str] = set()
    
    for topic, keywords in TOPIC_KEYWORDS.items():
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", lower_text):
                topics.add(topic)
                break
                
    if not topics:
        topics.add("general")
        
    return sorted(list(topics))

def extract_metadata(
    chunk: ArticleChunk,
    source_filename: str,
    default_date: Optional[str] = "2026-09-09",
    language: str = "english",
) -> Dict[str, Any]:
    """
    Builds the standardized metadata dictionary per context.md Section 7.
    """
    detected_date = extract_date_from_text(chunk.text) or default_date
    locations = detect_locations(chunk.text)
    topics = detect_topics(chunk.text)
    
    return {
        "text": chunk.text,
        "title": chunk.title,
        "source": source_filename,
        "page": chunk.page,
        "date": detected_date,
        "language": language,
        "location": locations,
        "topics": topics,
        "word_count": chunk.word_count,
    }
