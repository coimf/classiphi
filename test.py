import json
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tensorflow.keras.utils import pad_sequences
from sklearn import metrics
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification

def get_problems_and_topics():
    with open("remapped_training_data_1k.json", 'r') as file:
        data = json.load(file)
    df = pd.DataFrame(data)
    problems = df['problem'].values
    topics = df['topic'].values
    return problems, topics

def get_input_ids_and_attention_masks(tokenizer, problems):
    tokenized_problems = []
    input_ids = []
    for problem in problems:
        tokenized = tokenizer.tokenize(f"[CLS] {problem} [SEP]")
        tokenized_problems.append(tokenized)
        input_ids.append(tokenizer.convert_tokens_to_ids(tokenized))

    input_ids = pad_sequences(input_ids, maxlen=512, dtype='long', padding='post', truncating='post')
    attention_masks = []
    for ids in input_ids:
        mask = [float(id > 0) for id in ids]
        attention_masks.append(mask)

    return (
        input_ids,
        attention_masks
    )

def get_test_dataloader(attention_masks, input_ids, labels, seed, batch_size):
    _, x_test, _, y_test, _, test_masks = train_test_split(
        input_ids, labels, attention_masks, test_size=0.5, random_state=seed, stratify=labels)
    test_inputs = torch.tensor(np.array(x_test))
    test_masks = torch.tensor(np.array(test_masks))
    test_labels = torch.tensor(np.array(y_test))
    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
    return test_dataloader

def plot_training_stats(training_stats) -> None:
    fig = plt.figure(figsize=(12,6))
    plt.title('Metrics over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Value')

    plt.plot([stats['train_loss'] for stats in training_stats], label='Train Loss')
    plt.plot([stats['val_loss'] for stats in training_stats], label='Validation Loss')
    plt.plot([stats['val_accuracy'] for stats in training_stats], label='Validation Accuracy')
    plt.plot([stats['precision'] for stats in training_stats], label='Precision')
    plt.plot([stats['recall'] for stats in training_stats], label='Recall')
    plt.plot([stats['f1'] for stats in training_stats], label='F1')

    plt.legend()
    plt.show()

def test(model, test_dataloader, mode="topics", labels=None):
    model.eval()
    all_preds = []
    all_labels = []

    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels
            )
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()

        label_ids = b_labels.to('cpu').numpy()
        preds = np.argmax(logits, axis=1)
        all_preds.extend(preds)
        all_labels.extend(label_ids)

    accuracy = metrics.accuracy_score(all_labels, all_preds)
    precision = metrics.precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = metrics.recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = metrics.f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print("  accuracy: {:.2%}".format(accuracy))
    print("  precision: \033[36m{0:.2f}\033[0m".format(precision))
    print("  recall: \033[36m{0:.2f}\033[0m".format(recall))
    print("  f1: \033[36m{0:.2f}\033[0m".format(f1))

    if mode == "topics":
        plot_confusion_matrix_topics(all_labels, all_preds)
    else:
        plot_confusion_matrix_skills(mode, labels, all_labels, all_preds)

    return metrics.accuracy_score(all_labels, all_preds)

def plot_confusion_matrix_topics(y_true,y_predicted):
    cm = metrics.confusion_matrix(y_true, y_predicted)
    print("Plotting Confusion Matrix for topics")
    labels = ["algebra", "geometry", "number_theory", "combinatorics"]
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    fig = plt.figure(figsize=(7,6))
    res = sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')
    plt.yticks([0, 1, 2, 3], labels, va='center')
    plt.title('Confusion Matrix - Topic Classifier')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    plt.close()

def plot_confusion_matrix_skills(skill, labels, y_true, y_predicted):
    cm = metrics.confusion_matrix(y_true, y_predicted)
    print(f"Plotting {skill.capitalize()} Confusion Matrix")
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    fig = plt.figure(figsize=(15,12))
    res = sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')
    plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], labels, va='center')
    plt.title(f"Confusion Matrix - {skill.replace('_', ' ').capitalize()}")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    plt.close()

def main():
    global device, model, train_dataloader, validation_dataloader, test_dataloader, optimizer, epochs, total_steps
    import gc
    seed = 42
    batch_size = 8
    label_map = {0 : "algebra", 1 : "geometry", 2 : "number_theory", 3 : "combinatorics"}
    label_to_id = {v: k for k, v in label_map.items()}

    model_name = "models/topic_classifier_9900_epoch3_0805_23-10-17"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=4,
        output_attentions=False,
        output_hidden_states=False,
    )

    problems, topics = get_problems_and_topics()
    topics = np.array([label_to_id[t] for t in topics], dtype=np.int64)
    input_ids, attention_masks = get_input_ids_and_attention_masks(tokenizer, problems)
    _, _, test_dataloader = get_train_val_test_dataloader(attention_masks, input_ids, topics, seed, batch_size)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    model.to(torch.device(device))
    test(model, test_dataloader)

if __name__ == "__main__":
    main()
