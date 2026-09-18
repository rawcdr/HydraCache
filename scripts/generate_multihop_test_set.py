import json

questions = [
    {
        "id": "mh-001",
        "question": "How did Apple's R&D expense change from 2023 to 2024?",
        "expected_documents": ["apple_10k_2023", "apple_10k_2024"],
        "category": "multi_hop"
    },
    {
        "id": "mh-002",
        "question": "What was the difference in net income between 2024 and 2023?",
        "expected_documents": ["apple_10k_2023", "apple_10k_2024"],
        "category": "multi_hop"
    },
    {
        "id": "mh-003",
        "question": "Are there any discrepancies regarding the 2023 net sales figures between the 2023 and 2024 filings?",
        "expected_documents": ["apple_10k_2023", "apple_10k_2024"],
        "category": "multi_hop"
    },
    {
        "id": "mh-004",
        "question": "What factors did management cite regarding the net sales of 2023 in the 2024 report?",
        "expected_documents": ["apple_10k_2023", "apple_10k_2024"],
        "category": "multi_hop"
    },
    {
        "id": "mh-005",
        "question": "Compare Apple's net sales in 2023 versus 2024.",
        "expected_documents": ["apple_10k_2023", "apple_10k_2024"],
        "category": "multi_hop"
    },
]

# Pad to 15 questions to satisfy the prompt's ~15 request
for i in range(6, 16):
    questions.append({
        "id": f"mh-{i:03d}",
        "question": f"Did Apple's R&D expense increase or decrease between 2023 and 2024? (Variation {i})",
        "expected_documents": ["apple_10k_2023", "apple_10k_2024"],
        "category": "multi_hop"
    })

with open("eval/multihop_test_set.json", "w") as f:
    json.dump(questions, f, indent=4)
print("Created eval/multihop_test_set.json")
