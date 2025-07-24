import torch
import time
from transformers import BertForSequenceClassification, BertTokenizer

model_name = "models/topic_classifier"
LABEL_MAP = {0: "algebra", 1: "geometry", 2: "number theory", 3: "combinatorics"}
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)
device = "cpu"

model = model.to(torch.device(device))
model.eval()
print(f"Model loaded to {device}.\n")

while True:
    problem = input("Enter problem for topic classification: ")
    start_time = time.time()
    input_ids = tokenizer(problem, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**input_ids)
    logits = outputs.logits
    predicted_label_id = torch.argmax(logits, dim=-1).item()
    print("Predicted label", f"'{LABEL_MAP[predicted_label_id].capitalize()}'", f"in {time.time() - start_time:.2f}s\n")
