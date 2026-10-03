import pandas as pd
import re
import pickle
import logging
import os

import unicodedata

# Helper: pretty section header for terminal
def section(title):
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")


# Text cleaning 
def clean_resume(text):
    # Remove URLs
    text = re.sub(r'http\S+\s*', ' ', text)
    # Remove email addresses (noise in real-world resumes)
    text = re.sub(r'\S+@\S+\.\S+', ' ', text)
    # Remove phone numbers (noise in real-world resumes)
    text = re.sub(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', ' ', text)
    # Remove retweet/cc artifacts
    text = re.sub(r'\bRT\b|\bcc\b', ' ', text)
    # Remove hashtags
    text = re.sub(r'#\S+', '', text)
    # Remove mentions
    text = re.sub(r'@\S+', ' ', text)
    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7f]', ' ', text)
    # Remove punctuation and special characters (keep letters/numbers)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    # Collapse extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def normalize_resume_key(text):
    """Normalize case, Unicode composition, and whitespace; preserve punctuation."""
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def load_dataset(path="data/raw/resume_dataset.csv"):
    """Validate the source data and return one original row per duplicate group."""
    df = pd.read_csv(path, dtype={"Resume": "string", "Category": "string"})
    required = {"Resume", "Category"}
    missing_columns = required - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing_columns))}")

    invalid_resumes = df["Resume"].isna() | df["Resume"].str.strip().eq("")
    invalid_labels = df["Category"].isna() | df["Category"].str.strip().eq("")
    if invalid_resumes.any() or invalid_labels.any():
        raise ValueError(
            "Invalid dataset: "
            f"missing/blank resumes={int(invalid_resumes.sum())}, "
            f"missing/blank labels={int(invalid_labels.sum())}. "
            f"Resume row indices: {df.index[invalid_resumes].tolist()}; "
            f"label row indices: {df.index[invalid_labels].tolist()}"
        )
    if df.empty:
        raise ValueError("Dataset contains no rows.")

    # This key is only for duplicate detection, not the destructive ML cleaner.
    keys = df["Resume"].map(normalize_resume_key)
    label_counts = df.groupby(keys)["Category"].nunique()
    conflicting_keys = label_counts[label_counts > 1].index
    statistics = {
        "Rows": len(df),
        "Categories": df["Category"].nunique(),
        "Unique resume texts": keys.nunique(),
        "Redundant duplicate rows": int(keys.duplicated().sum()),
        "Conflicting duplicate labels": len(conflicting_keys),
    }
    section("Dataset Audit")
    for name, value in statistics.items():
        print(f"{name}: {value}")
        logging.info("%s: %s", name, value)

    # Choosing a label silently would hide contradictory ground truth.
    if len(conflicting_keys):
        conflicts = df.loc[keys.isin(conflicting_keys), ["Category"]]
        raise ValueError(
            f"Conflicting duplicate labels in {len(conflicting_keys)} group(s). "
            "Resolve these labels before training. Affected row indices/categories: "
            f"{conflicts['Category'].to_dict()}"
        )

    # Repeated resumes can otherwise appear in both training and test data.
    # Keep the first original row; never write back to the CSV, so the source
    # remains available for auditing and reproducing this decision.
    return df.loc[~keys.duplicated(keep="first")].copy()


def main():
    # Setup logging 
    os.makedirs("output", exist_ok=True)

    logging.basicConfig(
        filename="output/project.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logging.info("Training started")


    section("Resume Classification System — Training")

    df = load_dataset()
    logging.info("Dataset loaded and deduplicated successfully")
    section("Deduplicated Category Distribution")
    print(df['Category'].value_counts().to_string())

    df['cleaned_resume'] = df['Resume'].apply(clean_resume)

    logging.info("Text cleaning completed")


    # TF-IDF 
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Improved TF-IDF: sublinear_tf reduces impact of very frequent terms;
    # ngram_range captures bigrams for richer features; min_df filters rare noise
    tfidf = TfidfVectorizer(
        stop_words='english',
        sublinear_tf=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000
    )

    X = tfidf.fit_transform(df['cleaned_resume'])
    y = df['Category']

    section("TF-IDF Vectorization")
    print(f"  Feature matrix shape : {X.shape}")
    print(f"  Vocabulary size      : {len(tfidf.vocabulary_)}")
    print(f"  Classes              : {y.nunique()}")

    logging.info("TF-IDF vectorization completed")


    # Train-test split
    # stratify=y ensures each class is proportionally represented in both splits
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logging.info("Train-test split completed")


    # Evaluation helpers
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import numpy as np

    def print_model_results(name, y_true, y_pred):
        """Print a clean summary for a given model's predictions."""
        acc = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred)
        section(f"{name} — Results")
        print(f"  Accuracy : {acc * 100:.2f}%")
        print(f"\nClassification Report:\n{report}")
        return acc, report


    def save_confusion_matrix(name, y_true, y_pred, labels):
        """Save confusion matrix as a plain-text grid to output folder."""
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        filepath = os.path.join("output", f"confusion_matrix_{name.replace(' ', '_')}.txt")
        with open(filepath, "w") as f:
            f.write(f"Confusion Matrix — {name}\n")
            f.write(f"{'':35s}" + "  ".join(f"{l[:6]:>6}" for l in labels) + "\n")
            for i, row in enumerate(cm):
                f.write(f"{labels[i][:35]:35s}" + "  ".join(f"{v:>6}" for v in row) + "\n")
        print(f"  Confusion matrix saved → {filepath}")


    # Naive Bayes
    from sklearn.naive_bayes import MultinomialNB

    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)

    nb_pred = nb_model.predict(X_test)
    nb_accuracy, nb_report = print_model_results("Naive Bayes", y_test, nb_pred)

    # Save NB report
    with open("output/classification_report_nb.txt", "w") as f:
        f.write("Naive Bayes Accuracy: " + str(nb_accuracy) + "\n\n")
        f.write(nb_report)

    save_confusion_matrix("Naive_Bayes", y_test, nb_pred, sorted(y.unique()))

    logging.info("Naive Bayes model trained and evaluated")


    # Logistic Regression 
    from sklearn.linear_model import LogisticRegression

    # C=5 gives slightly less regularization; saga solver scales better with large vocab
    lr_model = LogisticRegression(max_iter=1000, C=5, solver='saga', n_jobs=-1)
    lr_model.fit(X_train, y_train)

    lr_pred = lr_model.predict(X_test)
    lr_accuracy, lr_report = print_model_results("Logistic Regression", y_test, lr_pred)

    save_confusion_matrix("Logistic_Regression", y_test, lr_pred, sorted(y.unique()))

    logging.info("Logistic Regression model trained")


    # LinearSVC (often best for high-dimensional text classification)
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV

    # Wrap in CalibratedClassifierCV so we can get probability estimates later
    svc_base = LinearSVC(max_iter=2000, C=1.0)
    svc_model = CalibratedClassifierCV(svc_base)
    svc_model.fit(X_train, y_train)

    svc_pred = svc_model.predict(X_test)
    svc_accuracy, svc_report = print_model_results("LinearSVC", y_test, svc_pred)

    save_confusion_matrix("LinearSVC", y_test, svc_pred, sorted(y.unique()))

    logging.info("LinearSVC model trained and evaluated")


    # Pick the best model to save
    scores = {
        "Naive Bayes": (nb_accuracy, nb_model),
        "Logistic Regression": (lr_accuracy, lr_model),
        "LinearSVC": (svc_accuracy, svc_model),
    }

    best_name, (best_acc, best_model) = max(scores.items(), key=lambda kv: kv[1][0])

    section("Model Comparison")
    for name, (acc, _) in scores.items():
        marker = "  ◀ best" if name == best_name else ""
        print(f"  {name:<25} {acc * 100:.2f}%{marker}")

    print(f"\n  Saving best model: {best_name}")

    # Save model
    os.makedirs("models", exist_ok=True)

    # Always save Naive Bayes (original behaviour preserved) — keep model.pkl as best model
    with open("models/model.pkl", "wb") as f:
        pickle.dump((best_model, tfidf), f)

    # Also save all models for reference
    with open("models/nb_model.pkl", "wb") as f:
        pickle.dump((nb_model, tfidf), f)

    with open("models/lr_model.pkl", "wb") as f:
        pickle.dump((lr_model, tfidf), f)

    with open("models/svc_model.pkl", "wb") as f:
        pickle.dump((svc_model, tfidf), f)

    # Save a consolidated summary report
    with open("output/classification_report.txt", "w") as f:
        f.write("=" * 60 + "\n")
        f.write("   Resume Classifier — Training Summary\n")
        f.write("=" * 60 + "\n\n")
        for name, (acc, _) in scores.items():
            f.write(f"{name:<25} Accuracy: {acc * 100:.2f}%\n")
        f.write(f"\nBest Model: {best_name} ({best_acc * 100:.2f}%)\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"\nClassification Report ({best_name}):\n\n")
        f.write(nb_report if best_name == "Naive Bayes" else
                lr_report if best_name == "Logistic Regression" else svc_report)

    print("\n  All models and reports saved successfully.")

    logging.info(f"Best model ({best_name}) and TF-IDF saved successfully")
    logging.info("Training completed")

    section("Training Complete")
    print(f"  Best model  : {best_name}")
    print(f"  Accuracy    : {best_acc * 100:.2f}%")
    print(f"  Model saved → models/model.pkl")
    print(f"  Reports     → output/\n")


if __name__ == "__main__":
    main()
