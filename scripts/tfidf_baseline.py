"""Lab 3A: TF-IDF + LinearSVC baseline."""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score

DATA_PATH = "data/raw/bayan_feedback.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    train = df[df["split"] == "train"]
    validation = df[df["split"] == "validation"]
    test = df[df["split"] == "test"]

    X_train = train["text"]
    y_train = train["topic"]
    X_validation = validation["text"]
    y_validation = validation["topic"]
    X_test = test["text"]
    y_test = test["topic"]

    print("Train:", len(X_train))
    print("Validation:", len(X_validation))
    print("Test:", len(X_test))
    print("Topics:", sorted(y_train.unique()))

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=100_000,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_validation_tfidf = vectorizer.transform(X_validation)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LinearSVC()
    model.fit(X_train_tfidf, y_train)

    validation_pred = model.predict(X_validation_tfidf)
    test_pred = model.predict(X_test_tfidf)

    validation_f1 = f1_score(
        y_validation, validation_pred, average="macro"
    )
    test_f1 = f1_score(
        y_test, test_pred, average="macro"
    )

    print(f"Validation macro-F1: {validation_f1:.4f}")
    print(f"Frozen test macro-F1: {test_f1:.4f}")


if __name__ == "__main__":
    main()
