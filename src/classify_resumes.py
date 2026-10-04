import os
import sys
import pickle
import pandas as pd
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.predict import hybrid_predict as predict_resume

from src.document_extraction import (
    DocumentExtractionError, SUPPORTED_EXTENSIONS, classification_text,
    clean_pdf_text, extract_text, extract_pdf, extract_docx, extract_txt, extract_image,
)
from src.resume_intelligence import parse_resume, parse_resume_file


# Helper: pretty section header for terminal
def section(title):
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")


def hybrid_predict(raw_text, model):
    """Keep the batch return shape while sharing single-resume inference."""
    category, confidence, _, method = predict_resume(raw_text, model)
    return category, confidence, method


if __name__ == "__main__":
    section("Resume Classification System — Batch Classify")

    # Load the complete raw-text pipeline
    print("\n  Loading trained model...")
    try:
        with open("models/model.pkl", "rb") as f:
            model = pickle.load(f)
        print("  ✓ Model loaded successfully")
    except FileNotFoundError:
        print("  ✗ ERROR: models/model.pkl not found. Run main.py first to train the model.")
        sys.exit(1)

    # Determine input source: folder of files OR legacy CSV
    INPUT_FOLDER = "data/input/resumes"    # drop PDF/DOCX/TXT files here
    LEGACY_CSV   = "data/input/resumes_to_classify.csv"

    results = []
    skipped = []

    files = []
    if os.path.isdir(INPUT_FOLDER):
        files = [f for f in os.listdir(INPUT_FOLDER)
                 if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS]

    if files:
        # Real-world mode: read resume files from a folder
        section(f"Reading {len(files)} Resume File(s)")

        for filename in sorted(files):
            filepath = os.path.join(INPUT_FOLDER, filename)
            try:
                profile = parse_resume_file(filepath)
            except DocumentExtractionError as error:
                print(f"  SKIPPED {filename}: {error}")
                skipped.append(filename)
                continue
            raw_text = classification_text(profile.raw_text, filepath)

            # Image-based PDFs give zero text — cannot classify
            if not raw_text.strip():
                ext = os.path.splitext(filename)[1].lower()
                if ext == ".pdf":
                    print(f"  ✗ {filename:<40} SKIPPED — image-based PDF (no text layer)")
                    print(f"    → Convert to text-based PDF or copy-paste the text manually.")
                else:
                    print(f"  ✗ {filename:<40} SKIPPED — could not extract text")
                skipped.append(filename)
                continue

            category, confidence, method = hybrid_predict(raw_text, model)

            conf_str   = f"{confidence:.1f}%" if confidence is not None else "N/A"
            method_str = f"[{method}]" if method == "keyword" else ""
            print(f"  ✓ {filename:<40} → {category:<30} {conf_str} {method_str}")

            results.append({
                "Filename":          filename,
                "Predicted_Category": category,
                "Confidence_%":      conf_str,
                "Method":            method,
            })

    elif os.path.isfile(LEGACY_CSV):
        # Legacy mode: CSV with Resume text column (original behaviour)
        print(f"\n  ⚠  No individual resume files found at {INPUT_FOLDER}")
        print(f"     Falling back to CSV mode using: {LEGACY_CSV}\n")

        df_in = pd.read_csv(LEGACY_CSV)

        if "Resume" not in df_in.columns:
            print("  ✗ ERROR: CSV must contain a 'Resume' column.")
            sys.exit(1)

        for _, row in df_in.iterrows():
            profile = parse_resume(str(row.get("Resume", "")))
            raw_text = profile.raw_text
            category, confidence, method = hybrid_predict(raw_text, model)

            conf_str   = f"{confidence:.1f}%" if confidence is not None else "N/A"
            method_str = f"[{method}]" if method == "keyword" else ""
            name = row.get("Name", "—")
            print(f"  ✓ {name:<20} → {category:<30} {conf_str} {method_str}")

            entry = {"Predicted_Category": category,
                     "Confidence_%": conf_str,
                     "Method": method}
            entry.update({k: v for k, v in row.items()})
            results.append(entry)

    else:
        print(f"\n  ✗ No resumes found.")
        print(f"     Option 1 (recommended): Place PDF/DOCX/TXT files in  {INPUT_FOLDER}/")
        print(f"     Option 2 (CSV):         Place resumes_to_classify.csv in  data/input/")
        sys.exit(1)


    if not results:
        section("Done")
        print(f"  No resumes could be classified.\n")
        sys.exit(0)

    # create output folder if it doesn't exist
    os.makedirs("output", exist_ok=True)

    # clear old classification results so old runs don't linger
    import glob
    for f in glob.glob("output/*_resumes.csv"):
        try:
            os.remove(f)
        except Exception:
            pass

    df_out = pd.DataFrame(results)

    # Summary: show a count per predicted category
    section("Prediction Summary")
    category_counts = df_out["Predicted_Category"].value_counts()
    for cat, count in category_counts.items():
        print(f"  {cat:<40} {count} resume(s)")

    if skipped:
        print(f"\n  ⚠  Skipped {len(skipped)} file(s) — image-based or unreadable:")
        for s in skipped:
            print(f"     • {s}")

    print(f"\n  Saving grouped resume files...")

    # group resumes by predicted category and save
    for category in df_out["Predicted_Category"].unique():
        category_df = df_out[df_out["Predicted_Category"] == category]
        filename    = category.replace(" ", "_") + "_resumes.csv"
        filepath    = os.path.join("output", filename)
        category_df.to_csv(filepath, index=False)
        print(f"  ✓ Saved {filepath}  ({len(category_df)} resume(s))")

    section("Done")
    print(f"  {len(results)} resume(s) classified  |  {len(skipped)} skipped → output/ folder\n")