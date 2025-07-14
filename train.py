import json
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import Dataset

with open("training_data.json", 'r') as file:
    data = json.load(file)
rows = []
for topic in data["topics"]:
    for skill in data["topics"][topic]:
        for problem in data["topics"][topic][skill]:
            entry = data["topics"][topic][skill][problem]
            if not entry['problem'] == "" and not entry['solution'] == "":
                rows.append({'problem':entry['problem'], 'topic':topic, 'skill':skill})

df = pd.DataFrame(rows)
df['topic'] = LabelEncoder().fit_transform(df['topic'])
df['skill'] = LabelEncoder().fit_transform(df['skill'])


model_name = "aieng-lab/math_pretrained_bert_mamut"
tokenizer = BertTokenizer.from_pretrained(model_name)
# training topics (not subcategories) for now
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(df['topic'].unique()))
print(model_name, "vocab size:", len(tokenizer))

ds = Dataset.from_pandas(df)

def tokenize_and_encode_labels(examples):
    tokenized = tokenizer(
        examples['problem'],
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=512
    )
    tokenized['labels'] = examples['topic']
    return tokenized
tokenized_dataset = ds.map(tokenize_and_encode_labels, batched=True)

split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = split_dataset['train']
test_dataset = split_dataset['test']

early_stopping = EarlyStoppingCallback(early_stopping_patience=2)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {
        "accuracy": acc,
        "f1": f1
    }

training_args = TrainingArguments(
    output_dir='./results',
    save_strategy='steps',
    eval_strategy='steps',
    eval_steps=50,
    save_steps=50,
    logging_steps=10,
    learning_rate=2e-5,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=4,
    weight_decay=0.01,
    save_total_limit=1,
    load_best_model_at_end=True,
    dataloader_num_workers=0,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping],
)


model.to(torch.device('mps'))
trainer.train()
