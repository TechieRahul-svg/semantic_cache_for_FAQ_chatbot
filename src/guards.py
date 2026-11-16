import re
from typing import Tuple, Set

# Negation lexicon covering standard English and Romanized Hinglish variants
NEGATION_WORDS: Set[str] = {
    # English
    "not", "no", "never", "none", "neither", "nor", "nothing", "nowhere",
    "cannot", "cant", "can't", "dont", "don't", "wont", "won't",
    "didnt", "didn't", "isnt", "isn't", "arent", "aren't", "without",
    # Hinglish & Hindi Romanized
    "nahi", "nhi", "nahin", "na", "mat", "bina", "kabhi nahi"
}

# Regex patterns for user-specific PII
PII_PATTERNS = {
    # Order IDs e.g. ORD-1234567, OD123456789, #9876543
    "order_id": re.compile(r"\b(?:ord|order|od)[-_]?[0-9]{5,15}\b|#[0-9]{5,10}\b", re.IGNORECASE),
    # Indian Mobile Numbers: +91 or 0 or bare 10-digit starting with 6-9
    "phone": re.compile(r"(?:\+91[\-\s]?|91[\-\s]?|0)?[6-9]\d{9}\b"),
    # Standard email addresses
    "email": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    # UPI IDs (e.g. user@okhdfcbank, name@upi)
    "upi_id": re.compile(r"[a-zA-Z0-9.\-_]{2,256}@[a-zA-Z]{2,64}")
}

def contains_pii(text: str) -> Tuple[bool, str]:
    """
    Checks if text contains user-specific personal data (Order ID, phone, email, UPI).
    Returns (True, reason) if PII is detected, else (False, "").
    """
    if not text:
        return False, ""
    
    for pii_type, pattern in PII_PATTERNS.items():
        match = pattern.search(text)
        if match:
            return True, f"Contains {pii_type}: '{match.group(0)}'"
    return False, ""

def has_negation(text: str) -> bool:
    """
    Detects if the text contains any English or Hinglish negation words.
    Uses regex word boundaries for accurate token matching.
    """
    if not text:
        return False
    
    tokens = set(re.findall(r"\b[a-zA-Z']+\b", text.lower()))
    return bool(tokens.intersection(NEGATION_WORDS))

def is_negation_conflict(query: str, cached_question: str) -> bool:
    """
    Checks if there is an asymmetric negation conflict between query and cached question.
    Returns True if one has negation and the other does not (polarity mismatch).
    """
    q_neg = has_negation(query)
    c_neg = has_negation(cached_question)
    return q_neg != c_neg
