import pickle
import re
import os
import sys

# PDF and DOCX support — accept real resume files in addition to pasted text
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
    Returns: (category, confidence_pct, top3_list, method_used)
    """
    cleaned    = clean_resume(raw_text)
    vector     = tfidf.transform([cleaned])
    ml_pred    = model.predict(vector)[0]
    confidence = 0.0
    probs      = None
    top3       = []

    if hasattr(model, "predict_proba"):
        probs      = model.predict_proba(vector)[0]
        confidence = max(probs) * 100
        classes    = model.classes_
        top3_idx   = probs.argsort()[-3:][::-1]
        top3       = [(classes[i], probs[i] * 100) for i in top3_idx]

    # Use keyword fallback when ML is not confident enough
    if confidence < CONFIDENCE_THRESHOLD:
        kw_pred, kw_score = keyword_predict(raw_text)
        if kw_score >= 2:
            return kw_pred, confidence, top3, "keyword"

    return ml_pred, confidence, top3, "ML"


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
        print("  ⚠  python-docx not installed. Run: pip install python-docx")
        return ""
    doc = Document(filepath)
    return " ".join([para.text for para in doc.paragraphs if para.text.strip()])


if __name__ == "__main__":
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

    print("\n  Supported inputs:")
    print("    • Paste resume text directly")
    print("    • Enter a file path to a .pdf, .docx, .txt, .png, .jpg")
    print("  Type 'exit' to quit.\n")


    # interactive loop
    while True:
        print("─" * 50)
        user_input = input("  Resume text or file path ▶  ").strip()

        # Exit command
        if user_input.lower() == "exit":
            print("\n  Exiting classifier. Goodbye!\n")
            break

        # Empty input
        if not user_input:
            print("  ⚠  Nothing entered. Please paste text or give a file path.\n")
            continue

        # Check if the input is a file path
        if os.path.isfile(user_input):
            ext = os.path.splitext(user_input)[1].lower()

            if ext == ".pdf":
                print(f"  Reading PDF: {user_input}")
                raw_text = extract_pdf(user_input)
            elif ext == ".docx":
                print(f"  Reading DOCX: {user_input}")
                raw_text = extract_docx(user_input)
            elif ext == ".txt":
                with open(user_input, "r", encoding="utf-8", errors="ignore") as f:
                    raw_text = f.read()
            elif ext in [".png", ".jpg", ".jpeg"]:
                print(f"  Reading Image: {user_input}")
                if not OCR_SUPPORT:
                    print("  ⚠  pytesseract not installed. OCR requires Tesseract.")
                    raw_text = ""
                else:
                    try:
                        img = Image.open(user_input)
                        raw_text = pytesseract.image_to_string(img)
                    except Exception as e:
                        print(f"  ⚠  OCR failed: {e}")
                        raw_text = ""
            else:
                print(f"  ⚠  Unsupported file type '{ext}'. Use .pdf, .docx, .txt or images\n")
                continue

            # Image-based PDF gives empty text — cannot classify
            cleaned_raw = clean_pdf_text(raw_text)
            if not cleaned_raw.strip():
                if ext == ".pdf":
                    print("  ✗ Could not extract text — this appears to be an image-based PDF.")
                    print("    → It was scanned/photographed. Convert it to a text-based PDF first.\n")
                else:
                    print("  ✗ Could not extract text from this file.\n")
                continue

            sample_resume = raw_text
        else:
            # Treat input as pasted resume text
            sample_resume = user_input

        prediction, confidence, top3, method = hybrid_predict(sample_resume, model, tfidf)

        method_note = "  [keyword fallback — ML confidence was low]" if method == "keyword" else ""

        print(f"\n  ┌─ Predicted Category : {prediction}")
        if confidence > 0:
            print(f"  │  ML Confidence      : {confidence:.1f}%")
        if method_note:
            print(f"  │  Note               :{method_note}")
        if top3:
            print(f"  │")
            print(f"  │  Top 3 Predictions (ML):")
            for cat, prob in top3:
                print(f"  │    {cat:<30} {prob:.1f}%")
        print(f"  └{'─' * 45}\n")