import pandas as pd
import pickle
import re
import os
import sys


# Helper: pretty section header for terminal
def section(title):
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")


# text cleaning function
def clean_resume(text):
    # Remove URLs
    text = re.sub(r'http\S+\s*', ' ', text)
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


# Minimum token count to consider a resume valid
MIN_TOKEN_COUNT = 10

def is_valid_resume(text):
    """Return True if the text has enough content to classify reliably."""
    if not text or not isinstance(text, str):
        return False
    tokens = text.split()
    return len(tokens) >= MIN_TOKEN_COUNT


section("Resume Classification System — Batch Classify")

# load model and vectorizer
print("\n  Loading trained model...")
try:
    with open("models/model.pkl", "rb") as f:
        model, tfidf = pickle.load(f)
    print("  ✓ Model loaded successfully")
except FileNotFoundError:
    print("  ✗ ERROR: models/model.pkl not found. Run main.py first to train the model.")
    sys.exit(1)

print("\n  Reading resumes...")

# read CSV file containing resumes
try:
    df = pd.read_csv("data/input/resumes_to_classify.csv")
except FileNotFoundError:
    print("  ✗ ERROR: data/input/resumes_to_classify.csv not found.")
    sys.exit(1)

# Validate 'Resume' column exists
if "Resume" not in df.columns:
    print("  ✗ ERROR: Input CSV must contain a 'Resume' column.")
    sys.exit(1)

total = len(df)
print(f"  Found {total} resume(s) in the file.")

# clean resumes
df["cleaned_resume"] = df["Resume"].apply(
    lambda x: clean_resume(str(x)) if isinstance(x, str) else ""
)

# Flag short/empty resumes that may produce unreliable predictions
df["_valid"] = df["cleaned_resume"].apply(is_valid_resume)
invalid_count = (~df["_valid"]).sum()

if invalid_count > 0:
    print(f"\n  ⚠  {invalid_count} resume(s) are too short or empty and may yield unreliable predictions.")

print("\n  Converting resumes to TF-IDF vectors...")

X = tfidf.transform(df["cleaned_resume"])

print("  Predicting job categories...")

predictions = model.predict(X)

# add predictions to dataframe
df["Predicted_Category"] = predictions

# Mark low-confidence entries where the resume was flagged as too short
df.loc[~df["_valid"], "Predicted_Category"] = "UNCERTAIN — Too short"

# remove internal helper columns (not needed for HR)
df = df.drop(columns=["cleaned_resume", "_valid"])

# create output folder if it doesn't exist
os.makedirs("output", exist_ok=True)


# Summary: show a count per predicted category
section("Prediction Summary")
category_counts = df["Predicted_Category"].value_counts()
for cat, count in category_counts.items():
    print(f"  {cat:<35} {count} resume(s)")

print(f"\n  Saving grouped resume files...")

# group resumes by predicted category
for category in df["Predicted_Category"].unique():

    category_df = df[df["Predicted_Category"] == category]

    filename = category.replace(" ", "_").replace("—", "-") + "_resumes.csv"

    filepath = os.path.join("output", filename)

    category_df.to_csv(filepath, index=False)

    print(f"  ✓ Saved {filepath}  ({len(category_df)} resume(s))")

section("Done")
print(f"  {total} resume(s) processed → output/ folder\n")