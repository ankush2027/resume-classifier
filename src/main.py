"""Reproducible resume-category baseline; the test set never selects the model."""
import hashlib
import json
import logging
import pickle
import platform
import shutil
import sys
import unicodedata
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

# Keep serialized preprocessing importable as src.preprocessing, including CLI use.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from src.preprocessing import clean_resume

SEED = 42
TFIDF_CONFIG = dict(stop_words="english", sublinear_tf=True,
                    ngram_range=(1, 2), min_df=2, max_features=50000)


def section(title):
    print(f"\n{'─' * 50}\n  {title}\n{'─' * 50}")


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
    deduplicated = df.loc[~keys.duplicated(keep="first")].copy()
    deduplicated.attrs["audit"] = statistics
    return deduplicated


def build_models():
    """Fresh pipelines: every fit learns its own vocabulary and IDF weights."""
    classifiers = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(C=5, solver="saga", max_iter=1000,
                                                   random_state=SEED),
        "LinearSVC": LinearSVC(C=1, max_iter=2000, random_state=SEED),
    }
    return {name: Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=clean_resume, **TFIDF_CONFIG)),
        ("classifier", classifier),
    ]) for name, classifier in classifiers.items()}


def split_dataset(df):
    train, test = train_test_split(df, test_size=0.2, random_state=SEED,
                                   stratify=df["Category"])
    overlap = set(train.Resume.map(normalize_resume_key)) & set(test.Resume.map(normalize_resume_key))
    if overlap:
        raise ValueError("Duplicate-key overlap between training and test data.")
    if set(train.Category) != set(df.Category) or set(test.Category) != set(df.Category):
        raise ValueError("Split does not cover all categories; inspect class counts.")
    if train.Category.value_counts().min() < 2:
        raise ValueError("Two-fold validation requires at least two training rows per category.")
    return train, test


def metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def run_experiment(dataset_path="data/raw/resume_dataset.csv"):
    """Select on training folds, then make one held-out prediction per run."""
    df = load_dataset(dataset_path)
    train, test = split_dataset(df)
    models = build_models()
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
    folds = list(cv.split(train.Resume, train.Category))
    validation = {}
    for name, pipeline in models.items():
        results = []
        for fit_indices, validation_indices in folds:
            fit_rows, validation_rows = train.iloc[fit_indices], train.iloc[validation_indices]
            fitted = clone(pipeline).fit(fit_rows.Resume, fit_rows.Category)
            results.append(metrics(validation_rows.Category, fitted.predict(validation_rows.Resume)))
        validation[name] = {"folds": results,
                            "mean": {key: sum(row[key] for row in results) / len(results)
                                     for key in results[0]}}

    # Descending macro-F1, accuracy, weighted-F1; alphabetical name breaks exact ties.
    selected = sorted(models, key=lambda name: (
        -validation[name]["mean"]["macro_f1"],
        -validation[name]["mean"]["accuracy"],
        -validation[name]["mean"]["weighted_f1"], name))[0]
    pipeline = clone(models[selected]).fit(train.Resume, train.Category)
    predictions = pipeline.predict(test.Resume)
    labels = sorted(df.Category.unique().tolist())
    original_count = len(pd.read_csv(dataset_path))
    metadata = {
        "dataset_sha256": hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest(),
        "original_row_count": original_count, "deduplicated_row_count": len(df),
        "duplicate_count": original_count - len(df), "category_count": len(labels),
        "conflicting_duplicate_labels": df.attrs["audit"]["Conflicting duplicate labels"],
        "train_count": len(train), "test_count": len(test), "duplicate_key_overlap": len(set(train.Resume.map(normalize_resume_key)) &
                                     set(test.Resume.map(normalize_resume_key))),
        "random_seed": SEED,
        "split": {"test_size": 0.2, "stratify": "Category",
                  "train_row_indices": train.index.tolist(), "test_row_indices": test.index.tolist(),
                  "train_category_counts": train.Category.value_counts().to_dict(),
                  "test_category_counts": test.Category.value_counts().to_dict()},
        "cv": {"n_splits": 2, "shuffle": True, "random_state": SEED,
               "fold_membership": [{"train_row_indices": train.iloc[a].index.tolist(),
                                    "validation_row_indices": train.iloc[b].index.tolist()}
                                   for a, b in folds]},
        "tfidf_configuration": TFIDF_CONFIG,
        "preprocessing": "src.preprocessing.clean_resume",
        "classifier_configurations": {name: model.named_steps["classifier"].get_params()
                                      for name, model in models.items()},
        "selection_rule": "mean macro-F1, then accuracy, then weighted-F1, then alphabetical name",
        "selected_model": selected, "validation_results": validation,
        "held_out_test_metrics": metrics(test.Category, predictions),
        "classification_report": classification_report(test.Category, predictions, labels=labels,
                                                        output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(test.Category, predictions, labels=labels).tolist(),
        "confusion_matrix_labels": labels,
        "python_version": platform.python_version(),
        "package_versions": {name: version(name) for name in
                             ["pandas", "numpy", "scipy", "scikit-learn", "joblib"]},
    }
    report = "Validation results (training-only 2-fold CV)\n"
    for name, result in validation.items():
        report += f"{name}: {json.dumps(result)}\n"
    report += f"\nSelected model: {selected}\nHeld-out test results\n"
    report += json.dumps(metadata["held_out_test_metrics"], indent=2) + "\n"
    report += classification_report(test.Category, predictions, labels=labels, zero_division=0)
    report += "\nSmall test set: these results are preliminary, not evidence of broad generalization.\n"
    return pipeline, metadata, report


def save_results(pipeline, metadata, report):
    """Archive previous artifacts before replacing them; never retrain on test."""
    Path("models").mkdir(exist_ok=True)
    Path("output").mkdir(exist_ok=True)
    previous = list(Path("models").glob("*.pkl"))
    previous += list(Path("output").glob("classification_report*.txt"))
    previous += list(Path("output").glob("confusion_matrix*.txt"))
    previous += list(Path("output").glob("confusion_matrix*.csv"))
    previous += list(Path("output").glob("experiment_metadata.json"))
    if previous:
        archive = Path("output/legacy") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        archive.mkdir(parents=True)
        (archive / "README.txt").write_text(
            "Archived previous artifacts. Pre-Stage-0 results may contain leakage and must not "
            "be presented as valid unseen-resume performance. See current experiment_metadata.json.\n")
        for path in previous:
            destination = archive / path.parent.name / path.name
            destination.parent.mkdir(exist_ok=True)
            shutil.move(str(path), destination)
    with Path("models/model.pkl").open("wb") as stream:
        pickle.dump(pipeline, stream)
    Path("output/experiment_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    Path("output/classification_report.txt").write_text(report)
    pd.DataFrame(metadata["confusion_matrix"], index=metadata["confusion_matrix_labels"],
                 columns=metadata["confusion_matrix_labels"]).to_csv("output/confusion_matrix.csv")


def main():
    pipeline, metadata, report = run_experiment()
    save_results(pipeline, metadata, report)
    print(report)
    print("Saved complete raw-text pipeline to models/model.pkl")


if __name__ == "__main__":
    main()
