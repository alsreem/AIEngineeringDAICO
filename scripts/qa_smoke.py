"""Lab 3B QA Smoke Test Implementation."""
import argparse
import json
from pathlib import Path
from transformers import pipeline

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-file",
        default="data/eval/qa_smoke_set.json",
        help="Path to QA smoke evaluation file.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    eval_file = Path(args.eval_file)
    
    if not eval_file.exists():
        print(f"Error: {eval_file} not found.")
        return

    with open(eval_file, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    print("--- Loading Pretrained Question Answering Pipeline ---")
    qa_pipeline = pipeline(
        "question-answering",
        model="deepset/roberta-base-squad2"
    )

    correct = 0
    total = 0

    articles = qa_data.get("data", [])
    for article in articles:
        for paragraph in article.get("paragraphs", []):
            context = paragraph.get("context", "")
            for qa in paragraph.get("qas", []):
                question = qa.get("question", "")
                answers = qa.get("answers", [])
                expected = answers[0]["text"] if answers else ""
                
                if context and question:
                    total += 1
                    res = qa_pipeline(question=question, context=context)
                    pred_answer = res.get("answer", "").strip()
                    print(f"Q: {question}")
                    print(f"Predicted: '{pred_answer}' | Expected: '{expected}'\n")
                    if expected and (expected.lower() in pred_answer.lower() or pred_answer.lower() in expected.lower()):
                        correct += 1

    acc = (correct / total) if total > 0 else 1.0
    print(f"QA Smoke Test Pass Rate: {acc:.2f} ({correct}/{total})")

if __name__ == "__main__":
    main()
