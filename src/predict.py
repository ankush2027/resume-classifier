import pickle
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.document_extraction import (
    DocumentExtractionError, classification_text, clean_pdf_text,
    extract_pdf, extract_docx, OCR_SUPPORT,
)
from src.resume_intelligence import parse_resume, parse_resume_file


# Helper: pretty section header for terminal
def section(title):
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")


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


def hybrid_predict(raw_text, model):
    """Predict raw text; keyword overrides have no model probability attached."""
    ml_pred = model.predict([raw_text])[0]
    confidence, top3 = None, []
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([raw_text])[0]
        confidence = float(max(probs) * 100)
        indices = probs.argsort()[-3:][::-1]
        top3 = [(model.classes_[i], float(probs[i] * 100)) for i in indices]
        if confidence < CONFIDENCE_THRESHOLD:
            kw_pred, kw_score = keyword_predict(raw_text)
            if kw_score >= 2:
                return kw_pred, None, [], "keyword"
    # No probabilities (e.g. LinearSVC) is not evidence of low confidence.
    return ml_pred, confidence, top3, "ML"


if __name__ == "__main__":
    # Load the complete raw-text pipeline
    section("Resume Role Classifier")
    print("\n  Loading model...")

    try:
        with open("models/model.pkl", "rb") as f:
            model = pickle.load(f)
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
            try:
                profile = parse_resume_file(user_input)
            except DocumentExtractionError as error:
                print(f"  Could not read resume: {error}")
                continue
            sample_resume = classification_text(profile.raw_text, user_input, clean_images=False)
            if not sample_resume.strip():
                print("  No readable resume text found.")
                continue
        else:
            profile = parse_resume(user_input)
            sample_resume = profile.raw_text

        prediction, confidence, top3, method = hybrid_predict(sample_resume, model)

        method_note = "  [keyword fallback — ML confidence was low]" if method == "keyword" else ""

        print(f"\n  ┌─ Predicted Category : {prediction}")
        if confidence is not None:
            print(f"  │  ML Confidence      : {confidence:.1f}%")
        else:
            print("  │  ML Confidence      : N/A")
        if method_note:
            print(f"  │  Note               :{method_note}")
        if top3:
            print(f"  │")
            print(f"  │  Top 3 Predictions (ML):")
            for cat, prob in top3:
                print(f"  │    {cat:<30} {prob:.1f}%")
        print(f"  └{'─' * 45}\n")