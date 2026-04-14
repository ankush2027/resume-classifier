import pickle
import re
import sys

# ─────────────────────────────────────────────
# Helper: pretty section header for terminal
# ─────────────────────────────────────────────
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


# Minimum token count for a meaningful prediction
MIN_TOKEN_COUNT = 10

def is_valid_text(text):
    """Return True if text has enough content for a reliable prediction."""
    return bool(text) and len(text.split()) >= MIN_TOKEN_COUNT


# load model + tfidf together
section("Resume Role Classifier")
print("\n  Loading model...")

try:
    with open("models/model.pkl", "rb") as f:
        model, tfidf = pickle.load(f)
    print("  ✓ Model loaded successfully")
except FileNotFoundError:
    print("  ✗ ERROR: models/model.pkl not found. Run main.py first to train the model.")
    sys.exit(1)


# interactive loop
print("\n  Paste resume text below and press Enter.")
print("  Type 'exit' to quit.\n")

while True:
    print("─" * 50)
    sample_resume = input("  Resume text ▶  ").strip()

    # Exit command
    if sample_resume.lower() == "exit":
        print("\n  Exiting classifier. Goodbye!\n")
        break

    # Edge case: empty input
    if not sample_resume:
        print("  ⚠  No text entered. Please paste resume content.\n")
        continue

    # Edge case: text too short to be a real resume
    if not is_valid_text(sample_resume):
        print(f"  ⚠  Text is too short (< {MIN_TOKEN_COUNT} words). Please provide more resume content.\n")
        continue

    cleaned = clean_resume(sample_resume)
    vector = tfidf.transform([cleaned])

    prediction = model.predict(vector)[0]

    # Show confidence probabilities if the model supports it
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vector)[0]
        confidence = max(probs) * 100
        # Top-3 predictions
        classes = model.classes_
        top3_idx = probs.argsort()[-3:][::-1]
        print(f"\n  ┌─ Predicted Category : {prediction}")
        print(f"  │  Confidence         : {confidence:.1f}%")
        print(f"  │")
        print(f"  │  Top 3 Predictions:")
        for idx in top3_idx:
            print(f"  │    {classes[idx]:<30} {probs[idx] * 100:.1f}%")
        print(f"  └{'─' * 45}\n")
    else:
        print(f"\n  ✓ Predicted Category: {prediction}\n")