import os
import re
import sys
import pickle
import pandas as pd

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


# text cleaning function
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


# Domain keyword dictionary for hybrid classification fallback
# Used when ML model confidence is below threshold
DOMAIN_KEYWORDS = {
    "Data Science":             ["machine learning", "data science", "deep learning", "tensorflow",
                                 "pytorch", "neural network", "nlp", "natural language", "computer vision",
                                 "data analysis", "predictive", "sklearn", "scikit", "kaggle",
                                 "statistics", "regression", "classification", "clustering", "feature engineering"],
    "Java Developer":           ["java", "spring boot", "spring framework", "hibernate", "maven",
                                 "gradle", "j2ee", "jsp", "servlet", "junit", "jvm", "microservices", "struts"],
    "Python Developer":         ["python", "django", "flask", "fastapi", "pip", "virtualenv",
                                 "celery", "pytest", "asyncio", "sqlalchemy", "pydantic"],
    "DevOps Engineer":          ["devops", "docker", "kubernetes", "jenkins", "ci cd", "ansible",
                                 "terraform", "aws", "azure", "gcp", "linux", "bash", "deployment pipeline",
                                 "infrastructure as code", "helm"],
    "Testing":                  ["manual testing", "test cases", "bug tracking", "quality assurance",
                                 "qa", "test plan", "jira", "defect", "regression testing",
                                 "black box", "white box", "uat"],
    "Automation Testing":       ["selenium", "test automation", "robot framework", "testng",
                                 "appium", "cypress", "playwright", "automated test"],
    "Web Designing":            ["html", "css", "javascript", "react", "angular", "vue", "ui ux",
                                 "figma", "photoshop", "responsive design", "bootstrap", "sass",
                                 "frontend", "web design", "wordpress"],
    "HR":                       ["human resources", "recruitment", "hiring", "onboarding", "payroll",
                                 "employee relations", "talent acquisition", "performance management",
                                 "hr policies", "staffing"],
    "Hadoop":                   ["hadoop", "hdfs", "mapreduce", "hive", "pig", "spark", "big data",
                                 "hbase", "yarn", "zookeeper", "kafka", "cloudera"],
    "Blockchain":               ["blockchain", "ethereum", "solidity", "smart contract", "cryptocurrency",
                                 "web3", "nft", "defi", "hyperledger", "bitcoin"],
    "ETL Developer":            ["etl", "data warehouse", "informatica", "talend", "ssis",
                                 "data pipeline", "data integration", "olap", "oltp", "pentaho"],
    "Database":                 ["sql", "mysql", "postgresql", "mongodb", "oracle", "nosql",
                                 "dba", "data modeling", "stored procedure", "database administration", "redis"],
    "Operations Manager":       ["operations", "supply chain", "logistics", "process improvement",
                                 "inventory management", "vendor management", "operations management"],
    "Mechanical Engineer":      ["mechanical", "cad", "solidworks", "autocad", "manufacturing",
                                 "production", "hvac", "thermodynamics", "fluid dynamics"],
    "Electrical Engineering":   ["electrical", "circuit design", "pcb", "embedded systems", "plc",
                                 "scada", "power systems", "vlsi", "microcontroller", "arduino"],
    "Civil Engineer":           ["civil engineering", "construction", "structural", "surveying",
                                 "concrete", "foundation", "site supervision", "estimation"],
    "Sales":                    ["sales", "business development", "revenue", "client relationship",
                                 "crm", "lead generation", "negotiation", "target achievement", "b2b"],
    "SAP Developer":            ["sap", "abap", "s4hana", "sap hana", "sap mm", "sap sd",
                                 "sap fi", "sap co", "sap basis", "fiori", "bapi"],
    "Health and fitness":       ["health", "fitness", "nutrition", "physiotherapy", "gym",
                                 "personal trainer", "wellness", "yoga", "sports", "rehabilitation"],
    "PMO":                      ["pmo", "project management", "pmp", "agile", "scrum", "prince2",
                                 "stakeholder management", "project planning", "risk management"],
    "Arts":                     ["art", "design", "creative", "photography", "graphic design",
                                 "illustration", "animation", "video editing", "content creation"],
    "Business Analyst":         ["business analyst", "requirement gathering", "brd", "process mapping",
                                 "use case", "wireframe", "gap analysis", "business requirements"],
    "DotNet Developer":         [".net", "c#", "asp.net", "mvc", "entity framework", "visual studio",
                                 "wcf", "blazor", "xamarin", "dotnet core"],
    "Network Security Engineer":["network security", "firewall", "penetration testing", "ethical hacking",
                                 "vulnerability assessment", "siem", "ssl", "vpn", "ids", "ips", "cybersecurity"],
    "Advocate":                 ["advocate", "lawyer", "legal", "litigation", "court", "counsel",
                                 "attorney", "bar council", "legal advice", "solicitor"],
}

# If ML confidence is below this %, use keyword scoring as fallback
CONFIDENCE_THRESHOLD = 60.0


def keyword_predict(text):
    """Score text against domain keywords and return best matching category."""
    text_lower = text.lower()
    scores = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        scores[domain] = score
    best_cat = max(scores, key=scores.get)
    return best_cat, scores[best_cat]


def hybrid_predict(raw_text, model, tfidf):
    """
    Hybrid prediction:
    - ML model runs first
    - If ML confidence >= threshold: use ML result
    - If ML confidence < threshold: use keyword scoring as fallback
    Returns: (category, confidence_pct, method_used)
    """
    cleaned = clean_resume(raw_text)
    vector  = tfidf.transform([cleaned])

    ml_pred    = model.predict(vector)[0]
    confidence = 0.0

    if hasattr(model, "predict_proba"):
        probs      = model.predict_proba(vector)[0]
        confidence = max(probs) * 100

    # Use keyword fallback when ML is not confident enough
    if confidence < CONFIDENCE_THRESHOLD:
        kw_pred, kw_score = keyword_predict(raw_text)
        if kw_score >= 2:   # at least 2 strong keyword matches
            return kw_pred, confidence, "keyword"

    return ml_pred, confidence, "ML"


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

    # load model and vectorizer
    print("\n  Loading trained model...")
    try:
        with open("models/model.pkl", "rb") as f:
            model, tfidf = pickle.load(f)
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

            category, confidence, method = hybrid_predict(raw_text, model, tfidf)

            conf_str   = f"{confidence:.1f}%" if confidence > 0 else "N/A"
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
            category, confidence, method = hybrid_predict(raw_text, model, tfidf)

            conf_str   = f"{confidence:.1f}%" if confidence > 0 else "N/A"
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