import os
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro', zero_division=0)
    acc = accuracy_score(labels, predictions)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

def main():
    model_name = "xlm-roberta-base"
    data_path = "data/raw/bayan_feedback.csv"
    
    df = pd.read_csv(data_path)
    text_col = 'text' if 'text' in df.columns else df.columns[0]
    label_col = 'topic' if 'topic' in df.columns else df.columns[-1]

    df = df.dropna(subset=[text_col, label_col]).reset_index(drop=True)

    # Mini subset for instant CPU completion (300 samples = ~18 steps)
    if len(df) > 300:
        df = df.sample(n=300, random_state=42).reset_index(drop=True)

    unique_labels = sorted(df[label_col].unique())
    num_labels = len(unique_labels)
    
    label2id = {label: int(i) for i, label in enumerate(unique_labels)}
    id2label = {int(i): label for i, label in enumerate(unique_labels)}
    df['labels'] = df[label_col].map(label2id).astype(int)

    raw_dataset = Dataset.from_pandas(df[[text_col, 'labels']])
    dataset_split = raw_dataset.train_test_split(test_size=0.2, seed=42)
    train_ds, eval_ds = dataset_split['train'], dataset_split['test']

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess_function(examples):
        return tokenizer(examples[text_col], max_length=32, truncation=True)

    train_ds = train_ds.map(preprocess_function, batched=True)
    eval_ds = eval_ds.map(preprocess_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, id2label=id2label, label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=5,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    print("Starting ultra-fast training...")
    trainer.train()

    print("Saving model...")
    os.makedirs("./models/classifier", exist_ok=True)
    model.save_pretrained("./models/classifier")
    tokenizer.save_pretrained("./models/classifier")
    print("Training finished successfully!")

if __name__ == "__main__":
    main()
