import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
from src.guards import has_negation, is_negation_conflict

data = json.load(open('results/raw_similarities.json', encoding='utf-8'))
flips = [d for d in data if d.get('negative_type') == 'negation_flip']

for f in flips:
    q = f['question']
    matched_q = f['top_1_faq_question']
    sim = f['top_1_similarity']
    q_has_neg = has_negation(q)
    matched_has_neg = has_negation(matched_q)
    conflict = is_negation_conflict(q, matched_q)
    print(f"Query ({'NEG' if q_has_neg else 'POS'}): '{q}'")
    print(f"   Matched ({'NEG' if matched_has_neg else 'POS'}): '{matched_q}' (sim={sim:.3f})")
    print(f"   Conflict: {conflict}\n")
