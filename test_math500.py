from transformers import BertForSequenceClassification, BertTokenizer
from datasets import load_dataset
import torch
import os

LABELS = {
    0: "algebra",
    1: "geometry",
    2: "number_theory",
    3: "combinatorics"
}

SUBJECT_TO_LABELS = {
    "Algebra": "algebra",
    "Intermediate Algebra": "algebra",
    "Prealgebra": "algebra",
    "Geometry": "geometry",
    "Number Theory": "number_theory",
    "Counting & Probability": "combinatorics"
}

def load_model(model_name: str, device_name: str = "cpu"):
    model_name = 'models/' + model_name
    device = torch.device(device_name)
    dtype = torch.float32 if device_name == "mps" else torch.float16
    model = BertForSequenceClassification.from_pretrained(model_name, torch_dtype=dtype)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(model_name)
    return model.to(device), tokenizer

def test(problem: str, expected: str, model, tokenizer) -> bool:
    encoded = tokenizer(
        problem,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    encoded = {k: v.to(model.device) for k, v in encoded.items()}

    with torch.inference_mode():
        logits = model(**encoded).logits

    pred_ids = logits.argmax(dim=-1).tolist()
    pred_labels = [LABELS[p].lower().strip() for p in pred_ids]
    pred_label = pred_labels[0]

    expected_label = SUBJECT_TO_LABELS[expected].lower().strip()
    correct = pred_label == expected_label
    if not correct:
        print(f"Problem: {problem}\nPredicted: {pred_label}, Expected: {expected_label}\n")
    return correct

def main():
    models = [m for m in os.listdir("models") if "topic" in m]
    print("\nid | model\n-----------------------------------------------------")
    for i, m in enumerate(models):
        print(f" {i+1} : {m}")
    model_id = -1
    while(model_id < 0 or model_id >= len(models)):
        model_id = int(input("\nEnter model id for testing: ")) - 1
    model, tokenizer = load_model(models[model_id], "mps")

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    ds = ds.filter(lambda x: x['subject'] in ['Algebra', 'Geometry', 'Counting & Probability', 'Number Theory', 'Intermediate Algebra', 'Prealgebra'])
    test_data = ds.to_pandas()

    num_correct = 0
    num_total = len(test_data)

    for row in test_data.itertuples():
        if test(row.problem, row.subject, model, tokenizer):
            num_correct += 1

    print(f"Accuracy: {num_correct}/{num_total} ({num_correct/num_total:.2%})")

if __name__ == "__main__":
    main()
