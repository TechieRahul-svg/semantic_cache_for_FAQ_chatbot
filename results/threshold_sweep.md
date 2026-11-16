# Semantic Cache Offline Threshold Sweep & Evaluation

## 1. Selected Threshold & Final Test Set Performance
- **Selected Threshold (tuned on Dev):** `T = 0.83`
- **5-Fold Cross Validation Thresholds:** `[np.float64(0.83), np.float64(0.83), np.float64(0.83), np.float64(0.8), np.float64(0.83)]` (Mean: `0.82`)

| Evaluation Split | Guard Mode | Hit Rate (Recall) | Precision | False Hits | 95% Wilson CI (Hit Rate) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Dev (60 queries)** | Baseline (No Guard) | 16.7% (6/36) | 100.0% | 0 | N/A (tuning set) |
| **Dev (60 queries)** | **With Negation Guard** | **16.7%** (6/36) | **100.0%** | **0** | N/A (tuning set) |
| **Test (40 queries)** | Baseline (No Guard) | 8.3% (2/24) | 100.0% | 0 | [0.08] |
| **Test (40 queries)** | **With Negation Guard** | **8.3%** (2/24) | **100.0%** | **0** | **[2.3%, 25.8%]** |

## 2. Negation Guard Trade-off on Dev Set
- **False Hits Prevented by Guard:** 0
- **Legitimate Hits Lost by Guard:** 0

### Key Caught Negatives (Prevented False Hits):