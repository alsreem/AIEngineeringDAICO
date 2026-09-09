"""Lab 3B solution: fine-tune token classification with correct alignment."""
import argparse
from pathlib import Path
import numpy as np
import evaluate
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="artifacts/ner",
        help="Where to save the trained NER artefact.",
    )
    return parser.parse_args()

def parse_conll(file_path):
    tokens_list, tags_list = [], []
    curr_tokens, curr_tags = [], []
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if curr_tokens:
                    tokens_list.append(curr_tokens)
                    tags_list.append(curr_tags)
                    curr_tokens, curr_tags = [], []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    curr_tokens.append(parts[0])
                    curr_tags.append(parts[-1])
        if curr_tokens:
            tokens_list.append(curr_tokens)
            tags_list.append(curr_tags)
            
    return tokens_list, tags_list

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conll_file = Path("data/models/bayan_ner.conll")
    if not conll_file.exists():
        print(f"Error: {conll_file} not found.")
        return

    tokens, tags = parse_conll(conll_file)
    
    unique_tags = sorted(list({t for seq in tags for t in seq}))
    if "O" not in unique_tags:
        unique_tags.append("O")
    label2id = {tag: i for i, tag in enumerate(unique_tags)}
    id2label = {i: tag for i, tag in enumerate(unique_tags)}

    model_name = "CAMeL-Lab/bert-base-arabic-camelbert-mix"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True, max_length=128
        )
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label2id[label[word_idx]])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    dataset = Dataset.from_dict({"tokens": tokens[:200], "ner_tags": tags[:200]})
    tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)
    split_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42)

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(unique_tags),
        id2label=id2label,
        label2id=label2id,
    )

    seqeval = evaluate.load("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [id2label[p_idx] for (p_idx, l_idx) in zip(prediction, label) if l_idx != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l_idx] for (p_idx, l_idx) in zip(prediction, label) if l_idx != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        max_steps=10,
        weight_decay=0.01,
        save_strategy="no",
        logging_steps=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )

    print("--- Starting Quick NER Model Training ---")
    trainer.train()

    print("--- Evaluating NER Model ---")
    eval_results = trainer.evaluate()
    print(f"\nFinal NER entity-F1 Score: {eval_results['eval_f1']:.4f}\n")

    # Fix contiguous tensors before saving
    for p in trainer.model.parameters():
        p.data = p.data.contiguous()

    trainer.save_model(str(output_dir))
    print(f"Model saved successfully to {output_dir}")

if __name__ == "__main__":
    main()
