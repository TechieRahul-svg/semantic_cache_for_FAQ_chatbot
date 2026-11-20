import re
from typing import Tuple, Set, List, Dict, Optional

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

# Policy subject keywords for domain-scoped duration matching
SUBJECT_LEXICON: Dict[str, Set[str]] = {
    "return": {
        "return", "returns", "returning", "returned", "exchange", "exchanges",
        "wapsi", "badalna"
    },
    "refund": {
        "refund", "refunds", "refunded", "reimbursement", "credit"
    },
    "delivery": {
        "deliver", "delivery", "deliveries", "delivers", "delivered", "shipping", "ship",
        "ships", "shipped", "dispatch", "arrive", "transit", "pahunchega"
    },
    "cancellation": {
        "cancel", "cancellation", "cancellations", "cancels", "cancelled", "canceling", "radd"
    },
    "warranty": {
        "warranty", "warranties", "guarantee", "guarantees"
    }
}

# Matches explicit time durations (e.g. "30 days", "7 business days", "24 hours", "2 weeks")
DURATION_REGEX = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(hours?|hrs?|business\s+days?|working\s+days?|days?|weeks?|months?)\b",
    re.IGNORECASE
)

# Explicit unit mapping preserving business days vs calendar days distinction
UNIT_MAPPING = {
    "hour": ("calendar_day", 1.0 / 24.0),
    "hours": ("calendar_day", 1.0 / 24.0),
    "hr": ("calendar_day", 1.0 / 24.0),
    "hrs": ("calendar_day", 1.0 / 24.0),
    "day": ("calendar_day", 1.0),
    "days": ("calendar_day", 1.0),
    "week": ("calendar_day", 7.0),
    "weeks": ("calendar_day", 7.0),
    "month": ("calendar_day", 30.0),  # Note: months use a 30-calendar-day approximation
    "months": ("calendar_day", 30.0),  # Note: months use a 30-calendar-day approximation
    "business day": ("business_day", 1.0),
    "business days": ("business_day", 1.0),
    "working day": ("business_day", 1.0),
    "working days": ("business_day", 1.0),
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

HARD_DELIMITERS: Set[str] = {".", "?", "!", ";"}

def _find_nearest_subject(text_lower: str, dur_start: int, dur_end: int) -> Optional[str]:
    """
    Finds the nearest unambiguous policy subject keyword to a duration within the same sentence.
    Hard boundaries ('.', '?', '!', ';') strictly prevent associating across sentences.
    Commas (',') act as soft separators: candidates in the local comma clause are preferred,
    while nearest-subject distance within the sentence is used when distinguishing subjects.
    Returns None if no subject is found within 60 characters or if association is ambiguous.
    """
    # 1. Determine hard sentence boundaries (., ?, !, ;)
    sent_start = 0
    for i in range(dur_start - 1, -1, -1):
        if text_lower[i] in HARD_DELIMITERS:
            sent_start = i + 1
            break

    sent_end = len(text_lower)
    for i in range(dur_end, len(text_lower)):
        if text_lower[i] in HARD_DELIMITERS:
            sent_end = i
            break

    # 2. Determine local comma clause boundaries within the sentence
    clause_start = sent_start
    for i in range(dur_start - 1, sent_start - 1, -1):
        if text_lower[i] == ",":
            clause_start = i + 1
            break

    clause_end = sent_end
    for i in range(dur_end, sent_end):
        if text_lower[i] == ",":
            clause_end = i
            break

    candidates = []
    for subject, keywords in SUBJECT_LEXICON.items():
        for kw in keywords:
            pattern = r"\b" + re.escape(kw) + r"\b"
            for m in re.finditer(pattern, text_lower):
                k_start, k_end = m.start(), m.end()

                # Strictly enforce hard sentence boundaries - do not cross sentences
                if k_start < sent_start or k_end > sent_end:
                    continue

                # Calculate base directional distance to the duration
                if k_end <= dur_start:
                    base_dist = dur_start - k_end
                elif k_start >= dur_end:
                    base_dist = k_start - dur_end
                    # If keyword follows duration with milestone prepositions ('of delivery', 'from dispatch'),
                    # apply distance penalty so preceding governing action verb takes precedence
                    prefix = text_lower[dur_end:k_start].strip()
                    if prefix.startswith("of") or prefix.startswith("from") or prefix.startswith("after"):
                        base_dist += 20
                else:
                    base_dist = 0

                if base_dist > 60:
                    continue

                in_comma_clause = (k_start >= clause_start and k_end <= clause_end)

                # Count commas between keyword and duration as soft penalty
                sep_text = (
                    text_lower[k_end:dur_start] if k_end <= dur_start
                    else text_lower[dur_end:k_start]
                )
                comma_count = sep_text.count(",")
                effective_dist = base_dist + (comma_count * 20)

                candidates.append((effective_dist, in_comma_clause, subject))

    if not candidates:
        return None

    # Prefer candidates in the same local comma clause if present
    clause_candidates = [c for c in candidates if c[1]]
    active_candidates = clause_candidates if clause_candidates else candidates

    # Find minimum effective distance
    min_dist = min(c[0] for c in active_candidates)

    # Find all unique subjects achieving the minimum effective distance
    nearest_subjects = {c[2] for c in active_candidates if c[0] == min_dist}

    # If exactly one unambiguous subject is closest, return it; otherwise return None (ambiguous)
    if len(nearest_subjects) == 1:
        return next(iter(nearest_subjects))

    return None

def extract_scoped_durations(text: str) -> List[Tuple[Optional[str], float, str]]:
    """
    Extracts duration constraints and strictly binds each to its nearest unambiguous local subject.
    Returns list of tuples: (subject, normalized_value, unit_type)
    subject is None if no policy subject keyword appears locally or if association is ambiguous.
    """
    if not text:
        return []

    results = []
    text_lower = text.lower()
    matches = list(DURATION_REGEX.finditer(text_lower))

    for m in matches:
        val = float(m.group(1))
        raw_unit = re.sub(r"\s+", " ", m.group(2).lower())
        unit_type, multiplier = UNIT_MAPPING.get(raw_unit, ("calendar_day", 1.0))
        normalized_val = round(val * multiplier, 2)

        # Strictly nearest unambiguous local subject - no whole-text fallback
        subject = _find_nearest_subject(text_lower, m.start(), m.end())

        results.append((subject, normalized_val, unit_type))

    return results

# Matches milestone prepositions immediately preceding a keyword,
# accommodating arbitrary trailing whitespace, commas, hyphens, and colons.
# Note: 'of' is intentionally excluded to avoid demoting governing policy subjects.
MILESTONE_PREP_PATTERN = re.compile(
    r"\b(after|following|from|since|post)\b[\s,;:\-]*$",
    re.IGNORECASE
)

def _extract_governing_question_subject(question: str) -> Optional[str]:
    """
    Extracts the single governing policy subject from a question.
    - Demotes keywords preceded by milestone prepositions ('after delivery', 'following delivery').
    - Returns None if no subject is found or if multiple competing non-milestone subjects exist.
    """
    if not question:
        return None

    text_lower = question.lower()
    governing_subjects = set()
    milestone_subjects = set()

    for subject, keywords in SUBJECT_LEXICON.items():
        for kw in keywords:
            pattern = r"\b" + re.escape(kw) + r"\b"
            for m in re.finditer(pattern, text_lower):
                k_start = m.start()
                prefix = text_lower[:k_start]
                if MILESTONE_PREP_PATTERN.search(prefix):
                    milestone_subjects.add(subject)
                else:
                    governing_subjects.add(subject)

    # Prioritize non-milestone action subjects over reference milestone anchors
    active_subjects = governing_subjects if governing_subjects else milestone_subjects

    # Only return if there is exactly one unambiguous governing policy subject
    if len(active_subjects) == 1:
        return next(iter(active_subjects))

    return None

def is_numerical_constraint_conflict(
    query: str,
    cached_question: str,
    cached_answer: str
) -> bool:
    """
    Evaluates whether an explicit duration constraint in the query contradicts
    the policy duration in the cached entry on the SAME policy subject and unit.

    Conservatively allows hits if:
    - Query contains no durations.
    - Cache contains no durations.
    - Subject association is ambiguous (None) on either side.
    - Units differ (business days vs calendar days) without direct safe conversion.
    """
    query_durations = extract_scoped_durations(query)
    if not query_durations:
        return False

    q_durations = extract_scoped_durations(cached_question)
    a_durations = extract_scoped_durations(cached_answer)

    # Resolve orphaned answer durations using the cached question's governing subject
    governing_subj = _extract_governing_question_subject(cached_question)
    if governing_subj:
        resolved_a_durations = []
        for subj, val, unit in a_durations:
            if subj is None:
                resolved_a_durations.append((governing_subj, val, unit))
            else:
                resolved_a_durations.append((subj, val, unit))
        a_durations = resolved_a_durations

    cache_durations = q_durations + a_durations
    if not cache_durations:
        return False

    # Compare durations ONLY when both sides have recognized, identical subjects and identical unit types
    for q_subj, q_val, q_unit in query_durations:
        if q_subj is None:
            # Ambiguous subject in query: conservative behavior -> do not block
            continue

        # Find cache durations for the exact same subject and unit
        matching_cache_vals = [
            c_val for c_subj, c_val, c_unit in cache_durations
            if c_subj == q_subj and c_unit == q_unit
        ]

        if not matching_cache_vals:
            # No duration constraint in cache for this specific subject and unit -> do not block
            continue

        # Check if query duration matches any allowed policy value for this subject (tolerance 0.1)
        compatible = any(abs(q_val - c_val) <= 0.1 for c_val in matching_cache_vals)
        if not compatible:
            return True  # Definite conflict on same subject and same unit -> reject hit

    return False
