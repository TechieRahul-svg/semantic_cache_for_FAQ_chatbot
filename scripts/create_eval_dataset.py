"""
Script to generate data/eval_dataset_100.json with exact 60 Dev / 40 Test stratification:
- Positives: 36 Dev, 24 Test (total 60: 2 per FAQ)
  - Language: ~60% EN, ~40% Hinglish
  - Legitimate negations: 6 (4 in Dev, 2 in Test)
- Negatives: 24 Dev, 16 Test (total 40)
  - 15 Same-topic, different-intent: 9 Dev, 6 Test
  - 10 Negation / opposite flips: 6 Dev, 4 Test
  - 15 Out-of-domain: 9 Dev, 6 Test
"""
import json

# 60 Positives: exactly 2 per FAQ across 30 FAQs.
# Split plan: for 18 FAQs, 1st is Dev, 2nd is Dev (18*2 = 36 Dev). Wait, better:
# For each FAQ, pick one dev and one test? Then it's 30 Dev, 30 Test.
# To get 36 Dev and 24 Test:
# 6 FAQs have both in Dev (6*2 = 12 Dev)
# 24 FAQs have 1 in Dev, 1 in Test (24 Dev, 24 Test)
# Total Dev = 12 + 24 = 36. Total Test = 24.
# That gives every single FAQ representation in Dev (30/30 FAQs covered), and 24 FAQs represented in Test!

positives_raw = [
    # faq_01 (Orders & Tracking: Track order)
    {"faq_id": "faq_01", "q": "Where is my package right now?", "lang": "en", "neg": False, "split": "dev"},
    {"faq_id": "faq_01", "q": "mera order track kaise karu kahan tak pahucha?", "lang": "hinglish", "neg": False, "split": "test"},
    
    # faq_02 (Orders & Tracking: Delivery delayed)
    {"faq_id": "faq_02", "q": "expected delivery date nikal gayi par parcel nahi mila", "lang": "hinglish", "neg": True, "split": "dev"},
    {"faq_id": "faq_02", "q": "My delivery date has already passed and order is still delayed", "lang": "en", "neg": False, "split": "test"},
    
    # faq_03 (Orders & Tracking: Address change)
    {"faq_id": "faq_03", "q": "Can I update my shipping address before dispatch?", "lang": "en", "neg": False, "split": "dev"},
    {"faq_id": "faq_03", "q": "order ship hone se pehle delivery address badal sakte hain kya?", "lang": "hinglish", "neg": False, "split": "test"},
    
    # faq_04 (Orders & Tracking: Express delivery)
    {"faq_id": "faq_04", "q": "How can I get next-day delivery on my order?", "lang": "en", "neg": False, "split": "dev"},
    {"faq_id": "faq_04", "q": "urgent delivery ya one day delivery ka option kaise milega?", "lang": "hinglish", "neg": False, "split": "dev"}, # Extra dev
    
    # faq_05 (Orders & Tracking: International shipping)
    {"faq_id": "faq_05", "q": "Do you deliver items to international locations outside India?", "lang": "en", "neg": False, "split": "dev"},
    {"faq_id": "faq_05", "q": "kya abroad me delivery karte ho India ke bahar?", "lang": "hinglish", "neg": False, "split": "test"},
    
    # faq_06 (Cancellations: Cancel before ship)
    {"faq_id": "faq_06", "q": "mujhe ye item nahi chahiye dispatch se pehle cancel kaise karein", "lang": "hinglish", "neg": True, "split": "dev"},
    {"faq_id": "faq_06", "q": "I want to cancel my pending order before it gets shipped", "lang": "en", "neg": False, "split": "test"},
    
    # faq_07 (Cancellations: Cancellation charges)
    {"faq_id": "faq_07", "q": "Will I be charged any penalty fee if I cancel my order?", "lang": "en", "neg": False, "split": "dev"},
    {"faq_id": "faq_07", "q": "order cancel karne pe koi charges ya cancellation fees lagti hai kya?", "lang": "hinglish", "neg": False, "split": "test"},
    
    # faq_08 (Cancellations: Cancel single item)
    {"faq_id": "faq_08", "q": "Can I cancel just one product from a combo order?", "lang": "en", "neg": False, "split": "dev"},
    {"faq_id": "faq_08", "q": "ek sath 3 item mangwaye the sirf ek item cancel karna hai", "lang": "hinglish", "neg": False, "split": "dev"}, # Extra dev
    
    # faq_09 (Payments & Pricing: COD availability)
    {"faq_id": "faq_09", "q": "How to check if cash on delivery is supported on my pin code?", "lang": "en", "neg": False, "split": "dev"},
    {"faq_id": "faq_09", "q": "mere area me cash on delivery COD mil jayega kya?", "lang": "hinglish", "neg": False, "split": "test"},
    
    # faq_10 (Payments & Pricing: Money debited order failed)
    {"faq_id": "faq_10", "q": "paise kat gaye account se par order confirm nahi hua", "lang": "hinglish", "neg": True, "split": "dev"},
    {"faq_id": "faq_10", "q": "Bank account was charged but order status says payment failed", "lang": "en", "neg": False, "split": "test"},
    
    # faq_11 (Payments & Pricing: EMI options)
    {"faq_id": "faq_11", "q": "What are the rules and eligible cards for no cost EMI?", "lang": "en", "neg": False, "split": "dev"},
    {"faq_id": "faq_11", "q": "no cost EMI pe saman lene ke liye kaunse credit cards chalte hain?", "lang": "hinglish", "neg": False, "split": "test"},
    
    # faq_12 (Payments & Pricing: Change payment method)
    {"faq_id": "faq_12", "q": "Can I switch payment method from COD to online payment after ordering?", "lang": "en", "neg": False, "split": "dev"},
    {"faq_id": "faq_12", "q": "placed order ka payment method online se cash me change ho sakta hai?", "lang": "hinglish", "neg": False, "split": "dev"}, # Extra dev
    
    # faq_13 (Returns & Exchanges: Return window)
    {"faq_id": "faq_13", "q": "How many days do I have to return an item after delivery?", "lang": "en", "neg": False, "split": "dev"},
    {"faq_id": "faq_13", "q": "delivery ke kitne din baad tak saman return kar sakte hain?", "lang": "hinglish", "neg": False, "split": "test"},
    
    # faq_14 (Returns & Exchanges: Schedule return pickup)
    {"faq_id": "faq_14", "q": "How do I raise a return request and schedule pickup?", "lang": "en", "neg": False, "split": "dev"},
    {"faq_id": "faq_14", "q": "delivered product ko wapas bhejne ke liye return request kaise dalein?", "lang": "hinglish", "neg": False, "split": "test"},
    
    # faq_15 (Returns & Exchanges: Pickup boy didn't arrive)
    {"faq_id": "faq_15", "q": "pickup boy nahi aaya parcel lene aaj ab kya karein?", "lang": "hinglish", "neg": True, "split": "dev"},
    {"faq_id": "faq_15", "q": "Courier delivery boy missed the scheduled return pickup appointment", "lang": "en", "neg": False, "split": "test"},
    
    # faq_16 (Returns & Exchanges: Size/color exchange)
    {"faq_id": "faq_16", "q": "Can I replace the delivered shirt with a larger size?", "lang": "en", "neg": False, "split": "dev"},
    {"faq_id": "faq_16", "q": "shoes ka size chota aa gaya bada size exchange kaise karein?", "lang": "hinglish", "neg": False, "split": "test"},
    
    # faq_17 (Returns & Exchanges: Damaged product)
    {"faq_id": "faq_17", "q": "Received a broken product inside the box, how to get replacement?", "lang": "en", "neg": False, "split": "dev"},
    {"faq_id": "faq_17", "q": "parcel open kiya to product damaged nikla replacement kaise milegi?", "lang": "hinglish", "neg": False, "split": "dev"}, # Extra dev
    
    # faq_18 (Returns & Exchanges: Missing item)
    {"faq_id": "faq_18", "q": "box khola toh charging cable nahi thi packet mein", "lang": "hinglish", "neg": True, "split": "dev"},
    {"faq_id": "faq_18", "q": "Package is missing an accessory that was supposed to be included", "lang": "en", "neg": False, "split": "test"},
    
    # faq_19 (Refunds: Refund timeline)
    {"faq_id": "faq_19", "q": "How much time does it take for return refund to hit my bank?", "lang": "en", "neg": False, "split": "dev"},
    {"faq_id": "faq_19", "q": "return parcel pickup hone ke baad refund kitne din me aayega?", "lang": "hinglish", "neg": False, "split": "test"},
    
    # faq_20 (Refunds: Refund delayed 7 days)
    {"faq_id": "faq_20", "q": "refund nahi aaya abhi tak 10 din ho gaye approved hue", "lang": "hinglish", "neg": True, "split": "test"}, # Legitimate negated in test
    {"faq_id": "faq_20", "q": "More than 7 business days have passed and refund has still not arrived", "lang": "en", "neg": True, "split": "dev"},
    
    # faq_21 (Refunds: COD refund to bank/UPI)
    {"faq_id": "faq_21", "q": "How do I provide my UPI ID or bank account for cash on delivery refund?", "lang": "en", "neg": False, "split": "dev"},
    {"faq_id": "faq_21", "q": "COD order ka refund bank account ya UPI me kaise transfer hoga?", "lang": "hinglish", "neg": False, "split": "test"},
    
    # faq_22 (Refunds: Wallet credit)
    {"faq_id": "faq_22", "q": "Can I choose store wallet credit for instant refund instead of bank?", "lang": "en", "neg": False, "split": "dev"},
    {"faq_id": "faq_22", "q": "bank me wait nahi karna kya refund turant wallet me le sakte hain?", "lang": "hinglish", "neg": False, "split": "dev"}, # Extra dev
    
    # faq_23 (Account & Invoices: Download invoice)
    {"faq_id": "faq_23", "q": "Where can I download the GST invoice copy for my past order?", "lang": "en", "neg": False, "split": "dev"},
    {"faq_id": "faq_23", "q": "purane order ka bill tax invoice download kahan se karein?", "lang": "hinglish", "neg": False, "split": "test"},
    
    # faq_24 (Account & Invoices: Update mobile/email)
    {"faq_id": "faq_24", "q": "How to edit the registered mobile number in my user profile?", "lang": "en", "neg": False, "split": "dev"},
    {"faq_id": "faq_24", "q": "account ka phone number aur email id change kaise karein?", "lang": "hinglish", "neg": False, "split": "test"},
    
    # faq_25 (Account & Invoices: Reset password)
    {"faq_id": "faq_25", "q": "Forgot my login password, how to reset it via OTP or email?", "lang": "en", "neg": False, "split": "dev"},
    {"faq_id": "faq_25", "q": "login password bhool gaya reset karne ka OTP kaise aayega?", "lang": "hinglish", "neg": False, "split": "test"},
    
    # faq_26 (Account & Invoices: Delete account)
    {"faq_id": "faq_26", "q": "How do I permanently delete my profile and shopping account?", "lang": "en", "neg": False, "split": "dev"},
    {"faq_id": "faq_26", "q": "apna shopping account permanently delete kaise karein?", "lang": "hinglish", "neg": False, "split": "test"},
    
    # faq_27 (Warranty & Services: Claim warranty)
    {"faq_id": "faq_27", "q": "How do I claim brand warranty with the official service center?", "lang": "en", "neg": False, "split": "dev"},
    {"faq_id": "faq_27", "q": "kharide hue gadget ki warranty brand center se claim kaise karein?", "lang": "hinglish", "neg": False, "split": "test"},
    
    # faq_28 (Warranty & Services: Authenticity)
    {"faq_id": "faq_28", "q": "Are the products sold on this platform 100 percent original and authentic?", "lang": "en", "neg": False, "split": "dev"},
    {"faq_id": "faq_28", "q": "kya platform pe bikne wale sare products genuine aur original hote hain?", "lang": "hinglish", "neg": False, "split": "dev"}, # Extra dev
    
    # faq_29 (Warranty & Services: Installation demo)
    {"faq_id": "faq_29", "q": "How to book installation and product demo for newly delivered TV?", "lang": "en", "neg": False, "split": "dev"},
    {"faq_id": "faq_29", "q": "delivered AC ya washing machine ka demo aur installation kaise hoga?", "lang": "hinglish", "neg": False, "split": "test"},
    
    # faq_30 (Payments & Pricing: Price match)
    {"faq_id": "faq_30", "q": "Will you refund the difference if the product price drops after I buy?", "lang": "en", "neg": False, "split": "dev"},
    {"faq_id": "faq_30", "q": "kya khareedne ke baad price kam hone par difference refund milega?", "lang": "hinglish", "neg": False, "split": "test"}
]

negatives_raw = [
    # 15 Same-topic, different-intent (9 Dev, 6 Test)
    {"faq_id": "none", "q": "Can I return an item after 30 days of purchase?", "lang": "en", "neg": False, "neg_type": "same_topic_diff_intent", "split": "dev"},
    {"faq_id": "none", "q": "Can I get my refund in physical cash instead of bank transfer or wallet?", "lang": "en", "neg": False, "neg_type": "same_topic_diff_intent", "split": "dev"},
    {"faq_id": "none", "q": "Can I cancel an order after the delivery executive has already delivered it?", "lang": "en", "neg": False, "neg_type": "same_topic_diff_intent", "split": "dev"},
    {"faq_id": "none", "q": "Can delivery agent bring package to my office address instead of home at delivery time?", "lang": "en", "neg": False, "neg_type": "same_topic_diff_intent", "split": "dev"},
    {"faq_id": "none", "q": "Can I pay COD order amount using Sodexo or gift vouchers at doorstep?", "lang": "en", "neg": False, "neg_type": "same_topic_diff_intent", "split": "dev"},
    {"faq_id": "none", "q": "Can I get a tax invoice with an overseas foreign company name and VAT number?", "lang": "en", "neg": False, "neg_type": "same_topic_diff_intent", "split": "dev"},
    {"faq_id": "none", "q": "Can I exchange an item for a completely different product category like shoes for a jacket?", "lang": "en", "neg": False, "neg_type": "same_topic_diff_intent", "split": "dev"},
    {"faq_id": "none", "q": "kya delivery boy se open box delivery inspect karwa ke accept kar sakte hain?", "lang": "hinglish", "neg": False, "neg_type": "same_topic_diff_intent", "split": "dev"},
    {"faq_id": "none", "q": "purana order delete karke order history clean kar sakte hain bina account band kiye?", "lang": "hinglish", "neg": False, "neg_type": "same_topic_diff_intent", "split": "dev"},
    
    {"faq_id": "none", "q": "Can I schedule a specific 2-hour morning time slot for return pickup?", "lang": "en", "neg": False, "neg_type": "same_topic_diff_intent", "split": "test"},
    {"faq_id": "none", "q": "Is the brand warranty applicable at authorized service centers outside India?", "lang": "en", "neg": False, "neg_type": "same_topic_diff_intent", "split": "test"},
    {"faq_id": "none", "q": "Can I convert an existing cash on delivery order into 6-month EMI after order placement?", "lang": "en", "neg": False, "neg_type": "same_topic_diff_intent", "split": "test"},
    {"faq_id": "none", "q": "kya return pickup ke time delivery boy ko cash dekar dusra product le sakte hain?", "lang": "hinglish", "neg": False, "neg_type": "same_topic_diff_intent", "split": "test"},
    {"faq_id": "none", "q": "kya sale ke baad price badh gaya toh purane rate pe dobara order kar sakte hain?", "lang": "hinglish", "neg": False, "neg_type": "same_topic_diff_intent", "split": "test"},
    {"faq_id": "none", "q": "kya return window expire hone ke baad bhi product return ho sakta hai?", "lang": "hinglish", "neg": False, "neg_type": "same_topic_diff_intent", "split": "test"},

    # 10 Negation / opposite polarity flips (6 Dev, 4 Test)
    {"faq_id": "none", "q": "order cancel mat karna galti se button press ho gaya tha please continue shipment", "lang": "hinglish", "neg": True, "neg_type": "negation_flip", "split": "dev"},
    {"faq_id": "none", "q": "mujhe refund nahi chahiye product replace kar do same item", "lang": "hinglish", "neg": True, "neg_type": "negation_flip", "split": "dev"},
    {"faq_id": "none", "q": "delivery boy aaya tha maine khud parcel lene se mana kiya tha", "lang": "hinglish", "neg": True, "neg_type": "negation_flip", "split": "dev"},
    {"faq_id": "none", "q": "return pickup cancel kaise karein mujhe product pasand aa gaya rakhna hai", "lang": "hinglish", "neg": False, "neg_type": "negation_flip", "split": "dev"},
    {"faq_id": "none", "q": "account delete mat karna temporary freeze ya deactivate karna hai", "lang": "hinglish", "neg": True, "neg_type": "negation_flip", "split": "dev"},
    {"faq_id": "none", "q": "parcel deliver mat karna main out of town hu agle hafte bhejo", "lang": "hinglish", "neg": True, "neg_type": "negation_flip", "split": "dev"},

    {"faq_id": "none", "q": "COD band kyu kar diya mere address pe pehle chalta tha ab nahi chal raha", "lang": "hinglish", "neg": True, "neg_type": "negation_flip", "split": "test"},
    {"faq_id": "none", "q": "maine koi return request nahi dali thi cancel my return request", "lang": "hinglish", "neg": True, "neg_type": "negation_flip", "split": "test"},
    {"faq_id": "none", "q": "I do not want to change my delivery address keep original default address", "lang": "en", "neg": True, "neg_type": "negation_flip", "split": "test"},
    {"faq_id": "none", "q": "mera refund aa gaya hai par extra discount amount cut hua hai", "lang": "hinglish", "neg": False, "neg_type": "negation_flip", "split": "test"},

    # 15 Truly Out-of-Domain (9 Dev, 6 Test)
    {"faq_id": "none", "q": "Who won the cricket match yesterday between India and Australia?", "lang": "en", "neg": False, "neg_type": "out_of_domain", "split": "dev"},
    {"faq_id": "none", "q": "Aaj Delhi aur NCR ka weather kaisa rahega barish hogi kya?", "lang": "hinglish", "neg": False, "neg_type": "out_of_domain", "split": "dev"},
    {"faq_id": "none", "q": "How to write a binary search algorithm in Python with recursion?", "lang": "en", "neg": False, "neg_type": "out_of_domain", "split": "dev"},
    {"faq_id": "none", "q": "What is the capital city of Australia?", "lang": "en", "neg": False, "neg_type": "out_of_domain", "split": "dev"},
    {"faq_id": "none", "q": "IRCTC portal pe tatkal train ticket booking time kya hota hai?", "lang": "hinglish", "neg": False, "neg_type": "out_of_domain", "split": "dev"},
    {"faq_id": "none", "q": "Suggest the best camera smartphone under 25000 in India currently", "lang": "en", "neg": False, "neg_type": "out_of_domain", "split": "dev"},
    {"faq_id": "none", "q": "SBI credit card reward points cash ya statement me redeem kaise karein?", "lang": "hinglish", "neg": False, "neg_type": "out_of_domain", "split": "dev"},
    {"faq_id": "none", "q": "How to apply for online passport renewal under tatkal scheme?", "lang": "en", "neg": False, "neg_type": "out_of_domain", "split": "dev"},
    {"faq_id": "none", "q": "Can you give me a high protein vegetarian meal plan for muscle building?", "lang": "en", "neg": False, "neg_type": "out_of_domain", "split": "dev"},

    {"faq_id": "none", "q": "Zomato gold membership subscription me restaurant dining discount kaise milta hai?", "lang": "hinglish", "neg": False, "neg_type": "out_of_domain", "split": "test"},
    {"faq_id": "none", "q": "What are the latest GST slabs and cess on luxury electric automobiles in India?", "lang": "en", "neg": False, "neg_type": "out_of_domain", "split": "test"},
    {"faq_id": "none", "q": "Parivahan se driving license renewal online slot kaise book karein?", "lang": "hinglish", "neg": False, "neg_type": "out_of_domain", "split": "test"},
    {"faq_id": "none", "q": "Can you explain the ending and timeline paradox of the movie Interstellar?", "lang": "en", "neg": False, "neg_type": "out_of_domain", "split": "test"},
    {"faq_id": "none", "q": "What is the current monthly and annual pricing for Netflix India premium plan?", "lang": "en", "neg": False, "neg_type": "out_of_domain", "split": "test"},
    {"faq_id": "none", "q": "Stock market Nifty index technical support and resistance levels for tomorrow", "lang": "en", "neg": False, "neg_type": "out_of_domain", "split": "test"}
]

dataset = []
idx = 1
for p in positives_raw:
    dataset.append({
        "id": f"eval_{idx:03d}",
        "question": p["q"],
        "faq_id": p["faq_id"],
        "split": p["split"],
        "language": p["lang"],
        "has_negation": p["neg"],
        "negative_type": None
    })
    idx += 1

for n in negatives_raw:
    dataset.append({
        "id": f"eval_{idx:03d}",
        "question": n["q"],
        "faq_id": "none",
        "split": n["split"],
        "language": n["lang"],
        "has_negation": n["neg"],
        "negative_type": n["neg_type"]
    })
    idx += 1

with open("data/eval_dataset_100.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

print(f"Generated {len(dataset)} evaluation questions.")
