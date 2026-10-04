import os
import re
import sys
import pickle
import pandas as pd
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.predict import hybrid_predict as predict_resume

# PDF and DOCX support — handles real-world resume files from job portals/email
try:
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    from docx import Document
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False

# OCR support for image-based PDFs and image files
try:
    import pytesseract
    from pdf2image import convert_from_path
    from PIL import Image
    OCR_SUPPORT = True
except ImportError:
    OCR_SUPPORT = False


# Helper: pretty section header for terminal
def section(title):
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")


# PDF-specific: clean up line-level noise from extraction
def clean_pdf_text(text):
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        line = line.strip()
        if re.match(r'^\d+$', line):   # standalone page numbers
            continue
        if len(line) < 3:              # too short to be meaningful
            continue
        cleaned.append(line)
    return ' '.join(cleaned)


def hybrid_predict(raw_text, model):
    """Keep the batch return shape while sharing single-resume inference."""
    category, confidence, _, method = predict_resume(raw_text, model)
    return category, confidence, method


# Extract text from a PDF file
def extract_pdf(filepath):
    if not PDF_SUPPORT:
        print("  ⚠  pdfplumber not installed. Run: pip install pdfplumber")
        return ""
    text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    
    cleaned = clean_pdf_text(text)
    
    # If standard extraction gets nothing, fallback to OCR
    if not cleaned.strip() and OCR_SUPPORT:
        try:
            images = convert_from_path(filepath)
            ocr_text = ""
            for img in images:
                ocr_text += pytesseract.image_to_string(img) + "\n"
            cleaned = clean_pdf_text(ocr_text)
        except Exception:
            pass

    return cleaned


# Extract text from a Word (.docx) file
def extract_docx(filepath):
    if not DOCX_SUPPORT:
        return ""
    doc = Document(filepath)
    return " ".join([para.text for para in doc.paragraphs if para.text.strip()])


# Extract text from a plain text file
def extract_txt(filepath):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# Extract text from an image
def extract_image(filepath):
    if not OCR_SUPPORT:
        print("  ⚠  pytesseract not installed. OCR requires Tesseract.")
        return ""
    try:
        img = Image.open(filepath)
        text = pytesseract.image_to_string(img)
        return clean_pdf_text(text)
    except Exception as e:
        print(f"  ⚠  OCR failed: {e}")
        return ""


# Route file to correct extractor based on extension
def extract_text(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        return extract_pdf(filepath)
    elif ext == ".docx":
        return extract_docx(filepath)
    elif ext == ".txt":
        return extract_txt(filepath)
    elif ext in [".png", ".jpg", ".jpeg"]:
        return extract_image(filepath)
    else:
        return ""


# Supported file types
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".png", ".jpg", ".jpeg"}


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
            raw_text = extract_text(filepath)

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
            raw_text = str(row.get("Resume", ""))
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