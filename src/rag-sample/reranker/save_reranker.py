from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_id = "BAAI/bge-reranker-v2-m3"
save_path = "./bge-reranker-v2-m3"

# Download and save locally
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)

