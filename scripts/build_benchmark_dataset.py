import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.guards import contains_pii

with open("data/eval_dataset_100.json", "r", encoding="utf-8") as f:
    items = json.load(f)

# Ensure consistent schema with pii_type
for item in items:
    if "pii_type" not in item:
        item["pii_type"] = None

synthetic_pii = [
    # 5 Order IDs (3 dev, 2 test)
    {
        "id": "eval_101",
        "question": "Where is my parcel for order ORD-10001?",
        "faq_id": "none",
        "split": "dev",
        "language": "en",
        "has_negation": False,
        "negative_type": "pii",
        "pii_type": "order_id"
    },
    {
        "id": "eval_102",
        "question": "mera package kab aayega order id OD-20002 hai",
        "faq_id": "none",
        "split": "dev",
        "language": "hinglish",
        "has_negation": False,
        "negative_type": "pii",
        "pii_type": "order_id"
    },
    {
        "id": "eval_103",
        "question": "Please check the refund status for order #30003.",
        "faq_id": "none",
        "split": "dev",
        "language": "en",
        "has_negation": False,
        "negative_type": "pii",
        "pii_type": "order_id"
    },
    {
        "id": "eval_104",
        "question": "Can you cancel my shipment for ORD-40004?",
        "faq_id": "none",
        "split": "test",
        "language": "en",
        "has_negation": False,
        "negative_type": "pii",
        "pii_type": "order_id"
    },
    {
        "id": "eval_105",
        "question": "kripya mere order ORD-50005 ka invoice bhej dijiye",
        "faq_id": "none",
        "split": "test",
        "language": "hinglish",
        "has_negation": False,
        "negative_type": "pii",
        "pii_type": "order_id"
    },

    # 5 Indian Phone Numbers (2 dev, 3 test)
    {
        "id": "eval_106",
        "question": "Please update delivery contact to +91 9876500001.",
        "faq_id": "none",
        "split": "dev",
        "language": "en",
        "has_negation": False,
        "negative_type": "pii",
        "pii_type": "phone"
    },
    {
        "id": "eval_107",
        "question": "delivery boy ko bolo 9876500002 pe call kare",
        "faq_id": "none",
        "split": "dev",
        "language": "hinglish",
        "has_negation": False,
        "negative_type": "pii",
        "pii_type": "phone"
    },
    {
        "id": "eval_108",
        "question": "Send SMS confirmation of delivery to 9876500003.",
        "faq_id": "none",
        "split": "test",
        "language": "en",
        "has_negation": False,
        "negative_type": "pii",
        "pii_type": "phone"
    },
    {
        "id": "eval_109",
        "question": "Call me at +91-9876500004 regarding damaged package.",
        "faq_id": "none",
        "split": "test",
        "language": "en",
        "has_negation": False,
        "negative_type": "pii",
        "pii_type": "phone"
    },
    {
        "id": "eval_110",
        "question": "mera alternate number 9876500005 note kar lijiye",
        "faq_id": "none",
        "split": "test",
        "language": "hinglish",
        "has_negation": False,
        "negative_type": "pii",
        "pii_type": "phone"
    },

    # 5 Emails (3 dev, 2 test)
    {
        "id": "eval_111",
        "question": "Send the tax invoice PDF to customer.one@example.com.",
        "faq_id": "none",
        "split": "dev",
        "language": "en",
        "has_negation": False,
        "negative_type": "pii",
        "pii_type": "email"
    },
    {
        "id": "eval_112",
        "question": "kripya bill copy rahul.kumar@example.com par mail karein",
        "faq_id": "none",
        "split": "dev",
        "language": "hinglish",
        "has_negation": False,
        "negative_type": "pii",
        "pii_type": "email"
    },
    {
        "id": "eval_113",
        "question": "Please update my registered email to user3@example.com.",
        "faq_id": "none",
        "split": "dev",
        "language": "en",
        "has_negation": False,
        "negative_type": "pii",
        "pii_type": "email"
    },
    {
        "id": "eval_114",
        "question": "Resend delivery confirmation to test.buyer@example.com.",
        "faq_id": "none",
        "split": "test",
        "language": "en",
        "has_negation": False,
        "negative_type": "pii",
        "pii_type": "email"
    },
    {
        "id": "eval_115",
        "question": "mera refund receipt support.user@example.com pe bhej do",
        "faq_id": "none",
        "split": "test",
        "language": "hinglish",
        "has_negation": False,
        "negative_type": "pii",
        "pii_type": "email"
    },

    # 5 UPI IDs (2 dev, 3 test)
    {
        "id": "eval_116",
        "question": "Process my refund directly to user1@okhdfcbank.",
        "faq_id": "none",
        "split": "dev",
        "language": "en",
        "has_negation": False,
        "negative_type": "pii",
        "pii_type": "upi_id"
    },
    {
        "id": "eval_117",
        "question": "mera bacha hua refund payment1@okaxis pe transfer kar do",
        "faq_id": "none",
        "split": "dev",
        "language": "hinglish",
        "has_negation": False,
        "negative_type": "pii",
        "pii_type": "upi_id"
    },
    {
        "id": "eval_118",
        "question": "Can you credit store cash to buyer3@upi?",
        "faq_id": "none",
        "split": "test",
        "language": "en",
        "has_negation": False,
        "negative_type": "pii",
        "pii_type": "upi_id"
    },
    {
        "id": "eval_119",
        "question": "Please initiate return refund to shopper4@okicici.",
        "faq_id": "none",
        "split": "test",
        "language": "en",
        "has_negation": False,
        "negative_type": "pii",
        "pii_type": "upi_id"
    },
    {
        "id": "eval_120",
        "question": "paise refund5@paytm vpa handle par bhej dijiye",
        "faq_id": "none",
        "split": "test",
        "language": "hinglish",
        "has_negation": False,
        "negative_type": "pii",
        "pii_type": "upi_id"
    }
]

# Verify all synthetic PII queries trigger contains_pii
for pii_entry in synthetic_pii:
    detected, reason = contains_pii(pii_entry["question"])
    assert detected, f"Failed to detect PII in {pii_entry['question']}"
    print(f"Verified PII: {pii_entry['id']} -> {reason}")

all_120 = items + synthetic_pii
assert len(all_120) == 120

# Save to data/benchmark_dataset_120.json
with open("data/benchmark_dataset_120.json", "w", encoding="utf-8") as f:
    json.dump(all_120, f, indent=2, ensure_ascii=False)

print("Saved data/benchmark_dataset_120.json successfully.")
