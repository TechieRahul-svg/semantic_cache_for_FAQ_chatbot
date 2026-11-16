import json
import os
import sys
import tempfile
import math

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_cache")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple

from src.embeddings import EmbeddingModel
from src.guards import is_negation_conflict, has_negation

def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Computes Wilson score confidence interval for a binomial proportion.
    Returns (center_estimate, lower_bound, upper_bound).
    """
    if total == 0:
        return 0.0, 0.0, 0.0
    
    # 1.96 for 95% CI
    z = 1.95996
    p = successes / total
    denominator = 1 + (z**2) / total
    centre_adjusted_probability = (p + (z**2) / (2 * total)) / denominator
    half_width = (z * math.sqrt((p * (1 - p) / total) + ((z**2) / (4 * (total**2))))) / denominator
    
    lower = max(0.0, centre_adjusted_probability - half_width)
    upper = min(1.0, centre_adjusted_probability + half_width)
    return p, lower, upper

def compute_raw_similarities(faqs_path: str, eval_path: str, embedder: EmbeddingModel) -> List[Dict[str, Any]]:
    with open(faqs_path, "r", encoding="utf-8") as f:
        faqs = json.load(f)
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_items = json.load(f)

    faq_questions = [f["question"] for f in faqs]
    faq_ids = [f["faq_id"] for f in faqs]
    
    print(f"Embedding {len(faqs)} canonical FAQs...")
    faq_vectors = embedder.encode(faq_questions) # (30, 384)

    eval_questions = [e["question"] for e in eval_items]
    print(f"Embedding {len(eval_items)} evaluation questions...")
    eval_vectors = embedder.encode(eval_questions) # (100, 384)

    # Dot product matrix (100, 30) -> cosine similarity
    sim_matrix = np.dot(eval_vectors, faq_vectors.T)

    raw_results = []
    for i, item in enumerate(eval_items):
        sims = sim_matrix[i]
        top_1_idx = int(np.argmax(sims))
        top_1_sim = float(sims[top_1_idx])
        top_1_faq_id = faq_ids[top_1_idx]
        top_1_faq_question = faq_questions[top_1_idx]

        neg_conflict = is_negation_conflict(item["question"], top_1_faq_question)

        raw_results.append({
            "id": item["id"],
            "question": item["question"],
            "ground_truth_faq_id": item["faq_id"],
            "split": item["split"],
            "language": item["language"],
            "negative_type": item["negative_type"],
            "has_negation": item["has_negation"],
            "top_1_faq_id": top_1_faq_id,
            "top_1_faq_question": top_1_faq_question,
            "top_1_similarity": round(top_1_sim, 4),
            "negation_conflict": neg_conflict
        })

    return raw_results

def evaluate_threshold(items: List[Dict[str, Any]], threshold: float, use_guard: bool = False) -> Dict[str, Any]:
    total_positives = sum(1 for item in items if item["ground_truth_faq_id"] != "none")
    total_negatives = sum(1 for item in items if item["ground_truth_faq_id"] == "none")

    hits = 0
    correct_hits = 0
    false_hits = 0
    false_hit_details = []
    guard_blocked_positives = []
    guard_blocked_negatives = []

    for item in items:
        sim = item["top_1_similarity"]
        is_above_threshold = sim >= threshold
        is_blocked_by_guard = use_guard and item["negation_conflict"]

        if is_above_threshold:
            if is_blocked_by_guard:
                if item["ground_truth_faq_id"] == item["top_1_faq_id"]:
                    guard_blocked_positives.append(item)
                else:
                    guard_blocked_negatives.append(item)
                continue # Guard blocked this hit

            hits += 1
            if item["ground_truth_faq_id"] == item["top_1_faq_id"]:
                correct_hits += 1
            else:
                false_hits += 1
                false_hit_details.append(item)

    hit_rate = (correct_hits / total_positives) if total_positives > 0 else 0.0
    precision = (correct_hits / hits) if hits > 0 else 1.0

    return {
        "threshold": round(threshold, 2),
        "use_guard": use_guard,
        "hits": hits,
        "correct_hits": correct_hits,
        "false_hits": false_hits,
        "hit_rate": round(hit_rate, 4),
        "precision": round(precision, 4),
        "guard_blocked_positives": guard_blocked_positives,
        "guard_blocked_negatives": guard_blocked_negatives,
        "false_hit_details": false_hit_details
    }

def main():
    os.makedirs("results", exist_ok=True)
    embedder = EmbeddingModel("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    raw_results = compute_raw_similarities("data/faqs_30.json", "data/eval_dataset_100.json", embedder)
    
    # Save raw scores for reproducibility
    with open("results/raw_similarities.json", "w", encoding="utf-8") as f:
        json.dump(raw_results, f, indent=2, ensure_ascii=False)
    print("Saved raw similarity scores to results/raw_similarities.json")

    dev_items = [r for r in raw_results if r["split"] == "dev"]
    test_items = [r for r in raw_results if r["split"] == "test"]

    thresholds = [round(t, 2) for t in np.arange(0.70, 0.99, 0.01)]

    dev_baseline_sweep = []
    dev_guard_sweep = []

    for t in thresholds:
        dev_baseline_sweep.append(evaluate_threshold(dev_items, t, use_guard=False))
        dev_guard_sweep.append(evaluate_threshold(dev_items, t, use_guard=True))

    # Save Dev sweep results as CSV
    csv_lines = [
        "threshold,baseline_hits,baseline_correct,baseline_false,baseline_hit_rate,baseline_precision,"
        "guard_hits,guard_correct,guard_false,guard_hit_rate,guard_precision,false_hits_removed,hits_lost"
    ]
    for b, g in zip(dev_baseline_sweep, dev_guard_sweep):
        fh_removed = b["false_hits"] - g["false_hits"]
        hits_lost = b["correct_hits"] - g["correct_hits"]
        csv_lines.append(
            f"{b['threshold']},{b['hits']},{b['correct_hits']},{b['false_hits']},{b['hit_rate']},{b['precision']},"
            f"{g['hits']},{g['correct_hits']},{g['false_hits']},{g['hit_rate']},{g['precision']},"
            f"{fh_removed},{hits_lost}"
        )
    with open("results/threshold_sweep.csv", "w", encoding="utf-8") as f:
        f.write("\n".join(csv_lines))

    # Find optimal threshold on Dev (Max hit rate where false hits == 0)
    # Check guard sweep first:
    zero_fh_candidates = [g for g in dev_guard_sweep if g["false_hits"] == 0]
    if not zero_fh_candidates:
        zero_fh_candidates = [min(dev_guard_sweep, key=lambda x: x["false_hits"])]
    optimal_guard_row = max(zero_fh_candidates, key=lambda x: x["hit_rate"])
    optimal_t = optimal_guard_row["threshold"]

    # Corresponding baseline row on Dev
    optimal_baseline_row = next(b for b in dev_baseline_sweep if b["threshold"] == optimal_t)

    # Now evaluate this chosen threshold on the unseen TEST set!
    test_baseline_eval = evaluate_threshold(test_items, optimal_t, use_guard=False)
    test_guard_eval = evaluate_threshold(test_items, optimal_t, use_guard=True)

    # Compute Wilson CIs on Test set
    # Test has 24 positives, 16 negatives
    hr_val, hr_low, hr_high = wilson_score_interval(test_guard_eval["correct_hits"], 24)
    prec_val, prec_low, prec_high = wilson_score_interval(test_guard_eval["correct_hits"], test_guard_eval["hits"])

    # 5-Fold Cross Validation on 100 items to check threshold stability
    print("Running 5-Fold Cross Validation for threshold stability...")
    np.random.seed(42)
    indices = np.random.permutation(len(raw_results))
    folds = np.array_split(indices, 5)
    cv_optimal_thresholds = []
    for fold_i in range(5):
        val_idx = set(folds[fold_i])
        train_items = [raw_results[k] for k in range(len(raw_results)) if k not in val_idx]
        # find best t on train
        t_scores = []
        for t in thresholds:
            ev = evaluate_threshold(train_items, t, use_guard=True)
            t_scores.append((ev["false_hits"], -ev["hit_rate"], t))
        best_t = min(t_scores)[2]
        cv_optimal_thresholds.append(best_t)

    # Save Negation Guard Catches & Trade-offs
    negation_catches_log = {
        "optimal_threshold": optimal_t,
        "dev_analysis": {
            "baseline_false_hits": optimal_baseline_row["false_hits"],
            "guard_false_hits": optimal_guard_row["false_hits"],
            "false_hits_removed": optimal_baseline_row["false_hits"] - optimal_guard_row["false_hits"],
            "hits_lost_by_guard": optimal_baseline_row["correct_hits"] - optimal_guard_row["correct_hits"],
            "examples_caught_false_hits": [
                {
                    "query": ex["question"],
                    "matched_faq": ex["top_1_faq_question"],
                    "similarity": ex["top_1_similarity"],
                    "type": ex["negative_type"]
                }
                for ex in optimal_guard_row["guard_blocked_negatives"][:5]
            ],
            "examples_lost_legitimate_hits": [
                {
                    "query": ex["question"],
                    "matched_faq": ex["top_1_faq_question"],
                    "similarity": ex["top_1_similarity"]
                }
                for ex in optimal_guard_row["guard_blocked_positives"][:5]
            ]
        },
        "test_analysis": {
            "test_guard_hit_rate": test_guard_eval["hit_rate"],
            "test_guard_precision": test_guard_eval["precision"],
            "test_guard_false_hits": test_guard_eval["false_hits"],
            "wilson_ci_hit_rate_95": [round(hr_low, 4), round(hr_high, 4)],
            "wilson_ci_precision_95": [round(prec_low, 4), round(prec_high, 4)]
        },
        "cv_5fold_thresholds": cv_optimal_thresholds
    }

    with open("results/negation_guard_catches.json", "w", encoding="utf-8") as f:
        json.dump(negation_catches_log, f, indent=2, ensure_ascii=False)

    # Plot Precision-Recall Curve
    plt.figure(figsize=(9, 5.5))
    recalls_b = [b["hit_rate"] for b in dev_baseline_sweep]
    precisions_b = [b["precision"] for b in dev_baseline_sweep]
    recalls_g = [g["hit_rate"] for g in dev_guard_sweep]
    precisions_g = [g["precision"] for g in dev_guard_sweep]

    plt.plot(recalls_b, precisions_b, "r--o", label="Dev Baseline (No Guard)", markersize=4)
    plt.plot(recalls_g, precisions_g, "b-s", label="Dev With Negation Guard", markersize=4)
    plt.scatter([optimal_guard_row["hit_rate"]], [optimal_guard_row["precision"]], 
                color="green", s=140, zorder=5, label=f"Selected Threshold T={optimal_t}")
    
    plt.title("Precision vs Hit Rate across Thresholds (Dev Set)", fontsize=13, fontweight='bold')
    plt.xlabel("Hit Rate (Recall on Positives)", fontsize=11)
    plt.ylabel("Hit Precision (Correct Hits / All Hits)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    plt.savefig("results/pr_curve.png", dpi=200)
    plt.close()

    # Generate Markdown Summary Table
    md_lines = [
        "# Semantic Cache Offline Threshold Sweep & Evaluation",
        "",
        "## 1. Selected Threshold & Final Test Set Performance",
        f"- **Selected Threshold (tuned on Dev):** `T = {optimal_t}`",
        f"- **5-Fold Cross Validation Thresholds:** `{cv_optimal_thresholds}` (Mean: `{np.mean(cv_optimal_thresholds):.2f}`)",
        "",
        "| Evaluation Split | Guard Mode | Hit Rate (Recall) | Precision | False Hits | 95% Wilson CI (Hit Rate) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
        f"| **Dev (60 queries)** | Baseline (No Guard) | {optimal_baseline_row['hit_rate']*100:.1f}% ({optimal_baseline_row['correct_hits']}/36) | {optimal_baseline_row['precision']*100:.1f}% | {optimal_baseline_row['false_hits']} | N/A (tuning set) |",
        f"| **Dev (60 queries)** | **With Negation Guard** | **{optimal_guard_row['hit_rate']*100:.1f}%** ({optimal_guard_row['correct_hits']}/36) | **{optimal_guard_row['precision']*100:.1f}%** | **{optimal_guard_row['false_hits']}** | N/A (tuning set) |",
        f"| **Test (40 queries)** | Baseline (No Guard) | {test_baseline_eval['hit_rate']*100:.1f}% ({test_baseline_eval['correct_hits']}/24) | {test_baseline_eval['precision']*100:.1f}% | {test_baseline_eval['false_hits']} | [{test_baseline_eval['hit_rate']:.2f}] |",
        f"| **Test (40 queries)** | **With Negation Guard** | **{test_guard_eval['hit_rate']*100:.1f}%** ({test_guard_eval['correct_hits']}/24) | **{test_guard_eval['precision']*100:.1f}%** | **{test_guard_eval['false_hits']}** | **[{hr_low*100:.1f}%, {hr_high*100:.1f}%]** |",
        "",
        "## 2. Negation Guard Trade-off on Dev Set",
        f"- **False Hits Prevented by Guard:** {optimal_baseline_row['false_hits'] - optimal_guard_row['false_hits']}",
        f"- **Legitimate Hits Lost by Guard:** {optimal_baseline_row['correct_hits'] - optimal_guard_row['correct_hits']}",
        "",
        "### Key Caught Negatives (Prevented False Hits):"
    ]
    for ex in optimal_guard_row["guard_blocked_negatives"][:3]:
        md_lines.append(f"- **Query:** *\"{ex['question']}\"*")
        md_lines.append(f"  - Matched FAQ: *\"{ex['top_1_faq_question']}\"* (Similarity: `{ex['top_1_similarity']}`)")
        md_lines.append(f"  - Result: Correctly BLOCKED by Negation Guard.")

    if optimal_guard_row["guard_blocked_positives"]:
        md_lines.append("\n### Legitimate Hits Lost (Cost of Guard):")
        for ex in optimal_guard_row["guard_blocked_positives"][:3]:
            md_lines.append(f"- **Query:** *\"{ex['question']}\"*")
            md_lines.append(f"  - Target FAQ: *\"{ex['top_1_faq_question']}\"* (Similarity: `{ex['top_1_similarity']}`)")
            md_lines.append(f"  - Cost: Blocked due to asymmetric negation polarity.")

    with open("results/threshold_sweep.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print("\n" + "\n".join(md_lines[:30]))
    print("\nGenerated results/threshold_sweep.csv, results/threshold_sweep.md, and results/pr_curve.png successfully.")

if __name__ == "__main__":
    main()
