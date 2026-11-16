import json

data = json.load(open('results/raw_similarities.json', encoding='utf-8'))
dev = [d for d in data if d['split'] == 'dev']

print(f"{'T':<6} | {'Baseline Hits':<14} | {'Base Correct':<12} | {'Base False':<10} | {'Guard Correct':<13} | {'Guard False':<11} | {'False Hits Blocked'}")
print("-" * 90)

for t in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.83, 0.85]:
    hits_b = [d for d in dev if d['top_1_similarity'] >= t]
    correct_b = [d for d in hits_b if d['ground_truth_faq_id'] == d['top_1_faq_id']]
    false_b = [d for d in hits_b if d['ground_truth_faq_id'] != d['top_1_faq_id']]
    
    hits_g = [d for d in hits_b if not d['negation_conflict']]
    correct_g = [d for d in hits_g if d['ground_truth_faq_id'] == d['top_1_faq_id']]
    false_g = [d for d in hits_g if d['ground_truth_faq_id'] != d['top_1_faq_id']]
    
    blocked = len(false_b) - len(false_g)
    print(f"{t:<6.2f} | {len(hits_b):<14} | {len(correct_b):<12}/36 | {len(false_b):<10} | {len(correct_g):<13}/36 | {len(false_g):<11} | {blocked}")

print("\n--- Example Caught False Hits at T=0.70 ---")
hits_70_b = [d for d in dev if d['top_1_similarity'] >= 0.70]
for fb in hits_70_b:
    if fb['ground_truth_faq_id'] != fb['top_1_faq_id']:
        status = "BLOCKED BY GUARD" if fb['negation_conflict'] else "PASSED GUARD"
        print(f"[{status}] (sim={fb['top_1_similarity']:.3f}) Q: '{fb['question']}'")
        print(f"   -> Matched: '{fb['top_1_faq_question']}' (Type: {fb['negative_type']})\n")
