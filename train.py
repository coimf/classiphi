import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

with open("training_data_1k.json", 'r') as file:
    data = json.load(file)
rows = []
for topic in data["topics"]:
    for skill in data["topics"][topic]:
        for problem in data["topics"][topic][skill]:
            entry = data["topics"][topic][skill][problem]
            if not entry['problem'] == "" and not entry['solution'] == "":
                rows.append({'problem':entry['problem'].replace('$', ''), 'topic':topic, 'skill':skill})

df = pd.DataFrame(rows)
df['topic'] = LabelEncoder().fit_transform(df['topic'])
df['skill'] = LabelEncoder().fit_transform(df['skill'])


model_name = "aieng-lab/math_pretrained_bert_mamut"
tokenizer = BertTokenizer.from_pretrained(model_name)
# training topics (not subcategories) for now
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=4)
# print(model_name, "vocab size:", len(tokenizer))

ds = Dataset.from_pandas(df)
def encode(examples):
    tokenized = tokenizer(
        examples['problem'],
        max_length=512,
        add_special_tokens=True,
        padding=True,
        truncation=True
    )
    tokenized['labels'] = examples['topic']
    return tokenized
tokenized_dataset = ds.map(encode, batched=True)

# print(tokenized_dataset[0])

split_dataset = tokenized_dataset.train_test_split(test_size=0.2)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {
        "accuracy": acc,
        "f1": f1
    }

print(split_dataset['train'])

# training_args = TrainingArguments(
#     output_dir='math-problem-classifier',
#     learning_rate=2e-5,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     num_train_epochs=6,
#     weight_decay=0.01,
#     save_total_limit=1,
#     load_best_model_at_end=True,
#     eval_strategy='epoch',
#     save_strategy='epoch',
#     gradient_accumulation_steps=2,
#     gradient_checkpointing=True,
#     logging_steps=10,
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=split_dataset['train'],
#     eval_dataset=split_dataset['test'],
#     compute_metrics=compute_metrics,
# )

# trainer.train()
