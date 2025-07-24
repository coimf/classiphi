import json
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from keras.utils import pad_sequences
from sklearn import metrics
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup


def get_combinatorics_problems_and_skills():
    with open("training_data_1k.json", 'r') as file:
        data = json.load(file)
    rows = []
    for skill in data["topics"]["combinatorics"]:
        for problem in data["topics"]["combinatorics"][skill]:
            entry = data["topics"]["combinatorics"][skill][problem]
            if not entry['problem'] == "" and not entry['solution'] == "":
                # no dolla signs
                rows.append({'problem': entry['problem'].replace(
                    '$', ''), 'skill': skill})

    df = pd.DataFrame(rows)
    problems = df['problem'].values
    skills = df['skill'].values
    return problems, skills


def get_input_ids_and_attention_masks(tokenizer, problems):
    tokenized_problems = []
    input_ids = []
    for problem in problems:
        tokenized = tokenizer.tokenize(f"[CLS] {problem} [SEP]")
        tokenized_problems.append(tokenized)
        input_ids.append(tokenizer.convert_tokens_to_ids(tokenized))

    input_ids = pad_sequences(input_ids, maxlen=512,
                              dtype='long', padding='post', truncating='post')
    # print(tokenized_problems[0])
    # print(input_ids[0])
    attention_masks = []
    for ids in input_ids:
        mask = [float(id > 0) for id in ids]
        attention_masks.append(mask)

    return (
        input_ids,
        attention_masks
    )


def get_train_val_test_dataloader(attention_masks, input_ids, topics, seed, batch_size):
    # first split: train+val vs test
    # train,val : test = 85 : 15
    x_train_val, x_test, y_train_val, y_test, mask_train_val, test_masks = train_test_split(
        input_ids, topics, attention_masks, test_size=0.15, random_state=seed)

    # second split: train vs val
    # (0.85-0.7)/0.85 = 0.1764705882
    # train : val : test = 70 : 15 : 15
    x_train, x_validation, y_train, y_validation, train_masks, validation_masks = train_test_split(
        x_train_val, y_train_val, mask_train_val, test_size=0.1765, random_state=seed)

    train_inputs = torch.tensor(np.array(x_train))
    validation_inputs = torch.tensor(np.array(x_validation))
    test_inputs = torch.tensor(np.array(x_test))

    train_masks = torch.tensor(np.array(train_masks))
    validation_masks = torch.tensor(np.array(validation_masks))
    test_masks = torch.tensor(np.array(test_masks))

    train_labels = torch.tensor(np.array(y_train))
    validation_labels = torch.tensor(np.array(y_validation))
    test_labels = torch.tensor(np.array(y_test))

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    # sample data randomly during training
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = TensorDataset(
        validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(
        validation_data)  # sample data sequentially
    validation_dataloader = DataLoader(
        validation_data, sampler=validation_sampler, batch_size=batch_size)

    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(
        test_data, sampler=test_sampler, batch_size=batch_size)

    return (
        train_dataloader,
        validation_dataloader,
        test_dataloader
    )


def train(model, train_dataloader, validation_dataloader, optimizer, epochs, total_steps):
    def flat_accuracy(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)

    # store training and validation loss, validation accuracy, timings
    training_loss = []
    validation_loss = []
    training_stats = []
    for epoch_i in range(0, epochs):
        print(f"Epoch {epoch_i+1} / {epochs}")
        print('Training model...')
        # reset total loss for this epoch
        total_train_loss = 0
        # put model in train mode
        model.train()

        for step, batch in enumerate(train_dataloader):
            # update progress every 40 batches.
            if step % 20 == 0 and not step == 0:
                print(f"  Batch {step} of {len(train_dataloader)}.")

            # As we unpack the batch, we'll also copy each tensor to the GPU using the `to` method. `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # clear gradients
            model.zero_grad()

            # forward pass (evaluate the model on this training batch).
            # It returns the loss (because we provided labels) and
            # the "logits"--the model outputs prior to activation.
            outputs = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels
            )
            loss = outputs[0]
            logits = outputs[1]
            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function returns the Python value
            # from the tensor.
            total_train_loss += loss.item()

            # backward pass to calculate the gradients.
            loss.backward()
            # Clip norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step along gradient
            optimizer.step()
            # Update lr
            scheduler.step()

        # Calculate the average loss over all batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        print("  avg_train_loss: {0:.2f}".format(avg_train_loss))
        # Validation
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("Evaluating on Validation Set")
        # Put model in eval mode
        model.eval()
        # Tracking variables
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0
        # Evaluate data for one epoch
        for batch in validation_dataloader:
            # Unpack validation batch from our dataloader.
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Tell pytorch not to construct the compute graph during forward pass, since this is only needed for backprop (training).
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                # "logits" are output values prior to applying activation function like softmax
                outputs = model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels
                )
                loss = outputs[0]
                logits = outputs[1]

            # accumulate validation loss
            total_eval_loss += loss.item()

            # move logits and labels to cpu
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # calculate accuracy for this batch of test sentences, and accumulate over all batches
            total_eval_accuracy += flat_accuracy(logits, label_ids)

        # report final accuracy for this validation run
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("Validation Accuracy: {0:.2f}".format(avg_val_accuracy))

        # calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)
        print("Validation Loss: {0:.2f}".format(avg_val_loss))

        training_loss.append(avg_train_loss)
        validation_loss.append(avg_val_loss)

        # record data from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_accuracy': avg_val_accuracy
            }
        )
    print("Training complete")
    return training_loss, validation_loss


def plot_losses(training_loss, validation_loss) -> None:
    fig = plt.figure(figsize=(12, 6))
    plt.title('Loss over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.plot(training_loss, label='Train')
    plt.plot(validation_loss, label='Validation')

    plt.legend()
    plt.show()


def test(model, test_dataloader):
    # print(f"Predicting labels for {len(test_input_ids)} test sentences...")

    # put model in eval mode
    model.eval()
    # tracking variables
    predictions = []
    true_labels = []

    # Predict
    for batch in test_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # unpack inputs from dataloader
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)

        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)

        # Final tracking variables
        y_logits, y_true, y_preds = [], [], []

        # Gather logit predictions
        for chunk in predictions:
            for logits in chunk:
                y_logits.append(logits)

        # Gather true labels
        for chunk in true_labels:
            for label in chunk:
                y_true.append(label)

        # Gather real predictions
        for logits in y_logits:
            y_preds.append(np.argmax(logits))

        print('Test Accuracy: {:.2%}'.format(
            metrics.accuracy_score(y_preds, y_true)))
        plot_confusion_matrix(y_true, y_preds)


def plot_confusion_matrix(y_true, y_predicted):
    cm = metrics.confusion_matrix(y_true, y_predicted)
    print("Plotting the Confusion Matrix")
    labels = [
        "Constructive Counting",
        "Complementary Counting",
        "Casework Counting",
        "Counting Independent Events",
        "Advanced Probability with Combinations",
        "Geometric Probability",
        "Counting with Restrictions",
        "Complementary Probability",
        "Expected Value",
        "Counting with Symmetry"
    ]
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    fig = plt.figure(figsize=(15, 12))
    res = sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')
    plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], labels, va='center')
    plt.title('Confusion Matrix - TestData')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    plt.close()


def main():
    global device
    seed = 42
    batch_size = 32
    label_map = {
        0: "Constructive Counting",
        1: "Complementary Counting",
        2: "Casework Counting",
        3: "Counting Independent Events",
        4: "Advanced Probability with Combinations",
        5: "Geometric Probability",
        6: "Counting with Restrictions",
        7: "Complementary Probability",
        8: "Expected Value",
        9: "Counting with Symmetry"
    }

    label_to_id = {v: k for k, v in label_map.items()}
    model_name = "aieng-lab/math_pretrained_bert_mamut"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    # training subcategories
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=10,
        output_attentions=False,
        output_hidden_states=False,
    )
    problems, skills = get_combinatorics_problems_and_skills()
    skills = np.array([label_to_id[t] for t in skills], dtype=np.int64)
    input_ids, attention_masks = get_input_ids_and_attention_masks(
        tokenizer, problems)
    train_dataloader, validation_dataloader, test_dataloader = get_train_val_test_dataloader(
        attention_masks, input_ids, skills, seed, batch_size)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    n_gpu = torch.cuda.device_count()
    model.to(torch.device(device))

    optimizer = AdamW(
        model.parameters(),
        lr=2e-5,
        eps=1e-8
    )

    epochs = 4
    total_steps = len(train_dataloader) * epochs

    training_loss, validation_loss = train(
        model, train_dataloader, validation_dataloader, optimizer, epochs, total_steps)

    plot_losses(training_loss, validation_loss)
    test(model, test_dataloader)

if __name__ == "__main__":
    main()
