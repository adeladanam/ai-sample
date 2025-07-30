import os

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "./../../../models/bge-reranker-v2-m3"  # local folder with model files
absolute_path = os.path.abspath(model_path)
print(absolute_path)

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)

query = "What is the capital of France?"
passages = [
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.",
    "Madrid is the capital of Spain."
]

# Prepare query-passage pairs
pairs = [(query, passage) for passage in passages]
inputs = tokenizer.batch_encode_plus(pairs, padding=True, truncation=True, return_tensors="pt")

# Run model
with torch.no_grad():
    scores = model(**inputs).logits.squeeze(-1)

# Sort results by score
ranked = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)
for i, (text, score) in enumerate(ranked):
    print(f"{i+1}. {text} (Score: {score.item():.4f})")
