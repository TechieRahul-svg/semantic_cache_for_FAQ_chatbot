import pytest
from src.guards import (
    is_negation_conflict,
    has_negation,
    contains_pii,
    extract_scoped_durations,
    is_numerical_constraint_conflict,
    SUBJECT_LEXICON,
    _extract_governing_question_subject
)

def test_rejects_30_day_return_against_7_day_policy():
    # Target eval_061 false-hit scenario
    query = "Can I return an item after 30 days of purchase?"
    cached_q = "How many days do I have to return an item after delivery?"
    cached_a = "Our return policy allows you to request a return within 7 days of delivery for eligible items."

    conflict = is_numerical_constraint_conflict(query, cached_q, cached_a)
    assert conflict is True, "30-day return query must be rejected when cached policy is 7 days"

def test_allows_matching_7_day_return_paraphrase():
    query = "Is it possible to return within 7 days of delivery?"
    cached_q = "What is the return window?"
    cached_a = "Returns are accepted within 7 days of receiving the package."

    conflict = is_numerical_constraint_conflict(query, cached_q, cached_a)
    assert conflict is False, "Matching 7-day return duration must be allowed"

def test_allows_7_business_days_versus_7_business_days():
    query = "My refund has not arrived after 7 business days, what should I do?"
    cached_q = "What if my refund is delayed after 7 business days?"
    cached_a = "If 7 business days have passed without bank credit, please contact support."

    conflict = is_numerical_constraint_conflict(query, cached_q, cached_a)
    assert conflict is False, "Matching business day durations must be allowed"

def test_does_not_equate_business_days_with_calendar_days():
    # 7 business days is NOT equivalent to 7 calendar days
    query = "Will I get refund in 7 days?"  # calendar days
    cached_q = "When is refund processed?"
    cached_a = "Refunds take 7 business days to reflect."  # business days

    # Different unit systems -> conservative: do not treat as identical unit match
    # extract_scoped_durations will label query as ('refund', 7.0, 'calendar_day')
    # and cache as ('refund', 7.0, 'business_day')
    q_dur = extract_scoped_durations(query)
    c_dur = extract_scoped_durations(cached_a)
    assert q_dur[0][2] == "calendar_day"
    assert c_dur[0][2] == "business_day"
    assert q_dur[0][2] != c_dur[0][2]

def test_normalizes_24_hours_to_1_calendar_day():
    query = "Can I cancel my shipment within 24 hours of ordering?"
    cached_q = "Cancellation policy"
    cached_a = "Orders can be cancelled within 1 day before dispatch."

    conflict = is_numerical_constraint_conflict(query, cached_q, cached_a)
    assert conflict is False, "24 hours should be treated as equivalent to 1 calendar day"

def test_allows_unrelated_durations():
    # Query mentions delivery duration, cache mentions return duration
    query = "Will my package arrive in 2 days?"
    cached_q = "Where can I return an item?"
    cached_a = "Returns are accepted within 7 days."

    conflict = is_numerical_constraint_conflict(query, cached_q, cached_a)
    assert conflict is False, "Delivery duration must not conflict with return duration"

def test_rejects_multiple_durations_with_one_return_conflict():
    query = "If delivery takes 2 days, can I return it after 30 days?"
    cached_q = "Return window"
    cached_a = "Returns are accepted within 7 days of delivery."

    conflict = is_numerical_constraint_conflict(query, cached_q, cached_a)
    assert conflict is True, "Conflict on return duration (30 vs 7) must trigger rejection"

def test_allows_open_ended_queries():
    query = "How many days do I have to return an item?"
    cached_q = "What is the return window?"
    cached_a = "Returns must be initiated within 7 days."

    conflict = is_numerical_constraint_conflict(query, cached_q, cached_a)
    assert conflict is False, "Open-ended query without numbers must never be blocked"

def test_ignores_non_duration_numbers():
    query = "Check status of order ORD-12345 price 500 rupees pin code 560001"
    cached_q = "Order status"
    cached_a = "Track order within 7 days."

    durations = extract_scoped_durations(query)
    assert len(durations) == 0, "Prices, PIN codes and order IDs must not be parsed as durations"

def test_ambiguous_subject_conservative_behavior():
    # Query mentions duration with no recognizable subject keyword in local window
    query = "Can this process be completed within 30 days?"
    cached_q = "Return window"
    cached_a = "Returns must be made within 7 days."

    conflict = is_numerical_constraint_conflict(query, cached_q, cached_a)
    assert conflict is False, "Ambiguous subject association must not block the cache hit"

def test_existing_negation_guard_preserved():
    # Affirmative vs negative polarity conflict
    assert is_negation_conflict("Can I cancel my order?", "Can I not cancel my order?") is True
    assert is_negation_conflict("Can I cancel my order?", "How to cancel my order?") is False
    assert has_negation("Item nahi chahiye") is True
    assert has_negation("Item chahiye") is False

def test_delivery_5_days_and_return_30_days_in_same_sentence():
    query = "Delivery takes 5 days, but I can return the item after 30 days."
    extracted = extract_scoped_durations(query)
    assert extracted == [
        ("delivery", 5.0, "calendar_day"),
        ("return", 30.0, "calendar_day")
    ]

def test_return_30_days_and_delivery_5_days_reversed_order():
    query = "I can return the item after 30 days, but delivery takes 5 days."
    extracted = extract_scoped_durations(query)
    assert extracted == [
        ("return", 30.0, "calendar_day"),
        ("delivery", 5.0, "calendar_day")
    ]

def test_two_subjects_within_same_local_context():
    # 'return' is the governing action verb, 'delivery' is a reference milestone
    query = "Our return policy allows you to request a return within 7 days of delivery for eligible items."
    extracted = extract_scoped_durations(query)
    assert len(extracted) == 1
    assert extracted[0] == ("return", 7.0, "calendar_day")

def test_ambiguous_subject_association_returns_none():
    # Equidistant conflicting keywords with no governing precedence
    # 'refund' and 'return' equidistant from '5 days' (distance 1 from either side)
    query = "return 5 days refund"
    extracted = extract_scoped_durations(query)
    assert len(extracted) == 1
    assert extracted[0][0] is None
    # Ambiguous subject should not cause a conflict
    assert is_numerical_constraint_conflict(query, "Return policy", "Returns accepted within 7 days.") is False

def test_delivery_5_days_period_returns_30_days():
    query = "Delivery takes 5 days. Returns are allowed within 30 days."
    extracted = extract_scoped_durations(query)
    assert extracted == [
        ("delivery", 5.0, "calendar_day"),
        ("return", 30.0, "calendar_day")
    ]

def test_duration_without_subject_does_not_inherit_from_preceding_sentence():
    query = "Delivery takes 5 days. You have 30 days."
    extracted = extract_scoped_durations(query)
    assert extracted == [
        ("delivery", 5.0, "calendar_day"),
        (None, 30.0, "calendar_day")
    ]

def test_paise_removed_from_refund_lexicon():
    assert "paise" not in SUBJECT_LEXICON.get("refund", set())
    assert "paisa" not in SUBJECT_LEXICON.get("refund", set())

def test_target_eval_061_false_hit_rejected_with_short_answer():
    # Exact eval_061 false-hit case: concise answer inherits 'return' from question
    query = "Can I return an item after 30 days of purchase?"
    cached_q = "How many days do I have to return an item after delivery?"
    cached_a = "Within 7 days."
    assert is_numerical_constraint_conflict(query, cached_q, cached_a) is True

def test_milestone_prepositions_robust_to_punctuation_and_whitespace():
    # Punctuation and whitespace variations in milestone detection
    q1 = "How many days do I have to return an item after delivery?"
    assert _extract_governing_question_subject(q1) == "return"

    q2 = "How many days do I have to return an item after, delivery?"
    assert _extract_governing_question_subject(q2) == "return"

    q3 = "How many days do I have to return an item following delivery?"
    assert _extract_governing_question_subject(q3) == "return"

    q4 = "How many days do I have to return an item post-delivery?"
    assert _extract_governing_question_subject(q4) == "return"

def test_competing_question_subjects_genuinely_detected_and_conservative():
    # Genuine competing subjects, including plural 'cancellations'
    assert "cancellations" in SUBJECT_LEXICON["cancellation"]
    assert "deliveries" in SUBJECT_LEXICON["delivery"]
    assert "warranties" in SUBJECT_LEXICON["warranty"]

    cached_q = "What is the policy for returns and cancellations?"
    assert _extract_governing_question_subject(cached_q) is None

    cached_a = "Within 7 days."
    query = "Can I return an item after 30 days?"
    assert is_numerical_constraint_conflict(query, cached_q, cached_a) is False

def test_answer_with_explicit_subjects_never_overwritten_by_question():
    # Explicit answer subject must not be overwritten
    cached_q = "Can I return an item?"
    cached_a = "Delivery takes 5 days."
    query = "Does delivery take 10 days?"
    # '5 days' in answer retains local subject 'delivery'
    assert is_numerical_constraint_conflict(query, cached_q, cached_a) is True

def test_business_day_unit_distinction_preserved_under_inheritance():
    # Business-day/calendar-day distinction preserved
    cached_q = "When will my refund be processed?"
    cached_a = "Within 7 business days."
    query = "Will my refund arrive in 7 days?"  # calendar days
    # Unit mismatch (calendar_day vs business_day) -> conservative allowance
    assert is_numerical_constraint_conflict(query, cached_q, cached_a) is False
