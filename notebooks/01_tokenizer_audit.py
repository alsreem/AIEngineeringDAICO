"""Lab 1: audit four tokenizer candidates on Bayan AR/EN text."""

from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

CANDIDATES = {
    "bert-base-multilingual-cased": "mBERT",
    "xlm-roberta-base": "XLM-R",
    "CAMeL-Lab/bert-base-arabic-camelbert-mix": "CAMeLBERT",
    "distilbert-base-uncased": "DistilBERT",
}

DATA = Path("data/raw/bayan_feedback.csv")


def fertility(tokenizer, texts) -> float:
    """Return total subword pieces divided by whitespace words."""
    total_pieces = 0
    total_words = 0

    for text in texts:
        words = text.split()
        if not words:
            continue

        pieces = tokenizer.tokenize(text)
        total_words += len(words)
        total_pieces += len(pieces)

    if total_words == 0:
        return 0.0

    return total_pieces / total_words


def main():
    df = pd.read_csv(DATA)

    results = []

    for model_name, label in CANDIDATES.items():
        print(f"\nLoading {label}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        for lang in ["ar", "en"]:
            texts = df.loc[df["lang"] == lang, "text"].dropna().astype(str).tolist()

            if not texts:
                continue

            fert = fertility(tokenizer, texts)

            lengths = [
                len(tokenizer.encode(text, add_special_tokens=True))
                for text in texts
            ]

            p95 = pd.Series(lengths).quantile(0.95)

            results.append(
                {
                    "Tokenizer": label,
                    "Language": lang,
                    "Samples": len(texts),
                    "Fertility": round(fert, 3),
                    "Mean length": round(sum(lengths) / len(lengths), 2),
                    "P95 length": round(p95, 2),
                }
            )

    result_df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("Bayan Tokenizer Audit")
    print("=" * 80)
    print(result_df.to_string(index=False))

    print("\nAudit complete.")


if __name__ == "__main__":
    main()
