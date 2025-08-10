from transformers import BertForSequenceClassification, BertTokenizer
from tqdm import tqdm
import torch
import json
import os

LABELS = {
    0: "algebra",
    1: "geometry",
    2: "number",
    3: "combinatorics"
}

def load_model(model_name: str, device_name: str = "cpu"):
    device = torch.device(device_name)
    dtype = torch.float32 if device_name == "mps" else torch.float16  # safer for MPS
    model = BertForSequenceClassification.from_pretrained(model_name, torch_dtype=dtype)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(model_name)
    return model.to(device), tokenizer

def batch_predict(problems, model, tokenizer, batch_size=1):
    """Run predictions in batches for speed."""
    predictions = []
    for i in tqdm(range(0, len(problems), batch_size), desc="Testing"):
        batch = problems[i:i + batch_size]
        batch = [p.replace('$', '') for p in batch]

        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        encoded = {k: v.to(model.device) for k, v in encoded.items()}

        with torch.inference_mode():
            logits = model(**encoded).logits

        pred_ids = logits.argmax(dim=-1).tolist()
        pred_labels = [LABELS[p].lower() for p in pred_ids]
        predictions.extend(pred_labels)
    return predictions

def main():
    models = [m for m in os.listdir("models") if "topic" in m]
    print("\nid | model\n-----------------------------------------------------")
    for i, m in enumerate(models):
        print(f" {i+1} : {m}")
    model_id = -1
    while(model_id < 0 or model_id >= len(models)):
        model_id = int(input("\nEnter model id for testing: ")) - 1

    test_sets = sorted(os.listdir("scraped_data/problems"))
    print("\nid | test set\n-----------------------------------------------------")
    for i, m in enumerate(test_sets):
        print(f" {i+1} : {m}")
    test_set_id = int(input("\nEnter test set id for testing (0 for all): ")) - 1

    if test_set_id < 0 or test_set_id >= len(test_sets):
        print("Testing all test sets. Check back in a bit.")
        if not os.path.exists("accuracies"):
            os.makedirs("accuracies")
        if not os.path.exists("failed_preds"):
            os.makedirs("failed_preds")
        accuracies = {}
        for test_set_id in tqdm(range(len(test_sets)), desc="Testing"):
            with open(f"scraped_data/problems/{test_sets[test_set_id]}", 'r') as f:
                test_data = json.load(f)
            problems = [f"{data['problem']} {data['answer_choices'] if 'frq' not in data['answer_choices'] else ''}" for data in test_data.values()]
            expected_label = test_sets[test_set_id].split("_")[1]
            expected_labels = [expected_label] * len(problems)
            model, tokenizer = load_model('models/'+models[model_id], "mps")
            predicted_labels = batch_predict(problems, model, tokenizer)
            correct_mask = [p == e for p, e in zip(predicted_labels, expected_labels)]
            num_correct = sum(correct_mask)
            num_total = len(problems)
            failed = []
            for prob, pred in zip(problems, predicted_labels):
                if pred != expected_label:
                    failed.append((prob, pred))
            with open(f"failed_preds/{test_sets[test_set_id].removesuffix('.json')}_failed.json", 'w') as f:
                json.dump(failed, f, indent=4)
            accuracies[test_sets[test_set_id]] = {
                "num_correct": num_correct,
                "num_total": num_total,
                "accuracy": num_correct / num_total
            }
        with open(f"accuracies/{models[model_id]}_{test_sets[test_set_id]}.json", 'w') as f:
            json.dump(accuracies, f, indent=4)

    with open(f"scraped_data/problems/{test_sets[test_set_id]}", 'r') as f:
        test_data = json.load(f)

    problems = [f"{data['problem']} {data['answer_choices'] if 'frq' not in data['answer_choices'] else ''}" for data in test_data.values()]
    expected_label = test_sets[test_set_id].split("_")[1]
    expected_labels = [expected_label] * len(problems)

    model, tokenizer = load_model('models/'+models[model_id], "mps")
    predicted_labels = batch_predict(problems, model, tokenizer)

    correct_mask = [p == e for p, e in zip(predicted_labels, expected_labels)]
    num_correct = sum(correct_mask)
    num_total = len(problems)

    failed = []
    for prob, pred in zip(problems, predicted_labels):
        if pred != expected_label:
            failed.append((prob, pred))

    with open(f"failed_preds/{test_sets[test_set_id].removesuffix('.json')}_failed.json", 'w') as f:
        json.dump(failed, f, indent=4)

    print(f"Accuracy: {num_correct}/{num_total} ({num_correct/num_total:.2%})")

if __name__ == "__main__":
    main()
