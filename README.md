# Resume Classification System 🚀

A machine learning system that automatically classifies real-world resumes into exactly **25 job categories**. 

Built to solve a real-world problem: companies receive resumes from multiple platforms (job portals, email, LinkedIn, etc.) in extremely different formats. This system handles them all automatically. It reads standard **PDFs, Word Documents (.docx), Text files (.txt), and even Image-based Scanned Resumes (.png, .jpg) via OCR!**

---

## 🌟 Key Features

- **Multi-Format Extraction:** Automatically extracts text from `.pdf`, `.docx`, and `.txt`.
- **Built-in OCR (Optical Character Recognition):** Can read scanned, photograph-based PDFs, `.jpg`, and `.png` image models via `Tesseract` and `pdf2image`.
- **Classification pipeline:** Uses TF-IDF with a classifier selected by training-only validation. Models with probabilities can use the existing keyword fallback below its threshold; LinearSVC returns an ML prediction with confidence shown as N/A. The keyword fallback is not part of the ML benchmark.
- **Smart Folder Batching:** Drop 100 random files in a folder, run one script, and get 25 beautifully organized folders sorted by job category.

---

## 🛠 Project Structure

```text
resume-classifier/
│
├── data/
│   ├── raw/
│   │   └── resume_dataset.csv          ← Main ML Training dataset (962 resumes)
│   └── input/
│       ├── resumes/                    ← DROP RESUMES HERE (.pdf, .png, .docx)
│       └── resumes_to_classify.csv     ← Fallback CSV if folder is empty!
│
├── src/
│   ├── main.py                         ← Train models, save best one (Run this first!)
│   ├── classify_resumes.py             ← Batch classify your 'input/resumes/' folder
│   └── predict.py                      ← Interactive single-resume terminal tool
│
├── models/                             ← Saved generated ML models
├── output/                             ← Automatically cleared & recreated results
├── requirements.txt
└── README.md
```

---

## 🚀 How To Install & Run

**If you are downloading this onto a new machine (Mac, Windows, or Linux)**, follow these exact steps to make sure Image/OCR scanning works perfectly:

### 1. Install System Requirements (For OCR Image Scanning)
To read scanned PDF images or pictures of resumes, this project relies on Poppler and Tesseract.

**Mac:**
```bash
brew install tesseract poppler
```

**Windows:**
1. Install [Tesseract OCR for Windows](https://github.com/UB-Mannheim/tesseract/wiki).
2. Download [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases/), extract it, and add the `Library\bin` or `bin` folder to your system's PATH.

*(Ubuntu: `sudo apt-get install tesseract-ocr poppler-utils`)*

### 2. Set Up Python Environment

**Mac / Linux:**
```bash
cd resume-classifier
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows:**
```powershell
cd resume-classifier
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
*(Note: `requirements.txt` should include `pdfplumber`, `python-docx`, `pytesseract`, `pdf2image`, `pandas`, `scikit-learn`, `pillow`.)*

### 3. Train the Model 🧠
You **must** run this first! It compares three pipelines using training-only two-fold cross-validation, selects by macro-F1, then evaluates once on held-out resumes and saves the complete pipeline. Legacy 100% results were affected by duplicate and TF-IDF leakage and are not valid unseen-resume performance.
```bash
python src/main.py   # Use python3 on Mac/Linux
```

### 4. Batch Classify Real Resumes 📂
**Option A (Folder Mode):** Simply drag and drop all your PDFs, Word files, and Images directly into the `data/input/resumes/` folder!
**Option B (CSV Mode):** If the folder is empty, the code dynamically falls back to the `resumes_to_classify.csv` file automatically.

Then, run:
```bash
python src/classify_resumes.py   # Use python3 on Mac/Linux
```
*Results will automatically generate beautifully separated CSVs in the `output/` folder.*

### 5. Check a Single File Interactively 🔍
Need to check just one candidate quickly? Run:
```bash
python src/predict.py   # Use python3 on Mac/Linux
```
It will open an interactive prompt. You can **paste a file path** (`/Users/You/resume.pdf` or `.png`) directly, or just copy and paste raw resume text to instantly see a breakdown of the ML prediction and accuracy!

### 6. Launch the Web App (Streamlit UI) 🌐
> ⚠️ **You must complete Step 3 (Train the Model) before this step**, otherwise the app will show an error because `models/model.pkl` does not exist yet.

```bash
streamlit run app.py
```
This opens a browser with the full drag-and-drop web interface for batch resume classification.

---

## 📌 Troubleshooting
- **`File does not exist` / `Invalid value` error in Streamlit:** The model hasn't been trained yet. Run `python src/main.py` first to generate `models/model.pkl`, then re-run `streamlit run app.py`.
- **`EmptyDataError` when running `main.py`:** Make sure your `data/raw/resume_dataset.csv` file actually has data in it and isn't 0 bytes! (You can type `git restore data/raw/resume_dataset.csv` if you accidentally cleared it).
- **File skipped because of "zlib / corrupted" error:** Sometimes downloaded PDFs are physically corrupted (zero text layer and compressed incorrectly). Open the file in an application like Mac's *Preview* app or a built-in PDF reader on Windows, select `Print` or `Export as PDF`, and save a fresh copy. Then the script will read it immediately.


## Stage 0 experiment

Use Python 3.9.6 with the pinned ML dependencies. Run from the repository root:

```bash
python -m unittest discover -s tests -v
python src/main.py
```

The raw CSV stays unchanged. Duplicate-key normalization is separate from model
preprocessing. After deduplication, raw text is split 80/20 with seed 42 and category
stratification. Each two-fold validation fit learns its own TF-IDF vocabulary.
Selection uses mean macro-F1, then accuracy, then weighted-F1, then alphabetical
model name. Only the selected pipeline is refitted on training data and evaluated
on test; test data is never used for fitting or selection.

`output/experiment_metadata.json` records dataset hash, original row indices for
splits/folds, settings, versions and full metrics. `output/classification_report.txt`
and `output/confusion_matrix.csv` describe the current experiment. Previous models
and reports are archived under `output/legacy/` before replacement.

`models/model.pkl` accepts raw strings and requires this repository's `src` package
on the Python import path. Run consumers from the repository root as above. LinearSVC
has no probabilities: confidence is displayed as N/A and this alone never triggers
keyword fallback. Keyword overrides also display N/A and are outside the ML benchmark.

There are few unique resumes per category; a small held-out result is preliminary.
Exact/normalized duplicate separation does not establish independence of near-duplicate
templates or prove real-world candidate performance. Tests repeat the fixed experiment
only to verify reproducibility, not to tune against test results.

## Stage 1: Resume document intelligence

```text
Resume file → shared document extraction → CandidateProfile → existing ML classification
```

The public API runs locally and deterministically, without an LLM or external API:

```python
from src.resume_intelligence import parse_resume, parse_resume_file

profile = parse_resume_file("resume.pdf")  # PDF, DOCX, TXT, PNG, JPG, JPEG
# Or: profile = parse_resume("Name: Asha Rao\nSkills: Python, PostgreSQL")
print(profile.to_dict())
```

`CandidateProfile` contains name, email, phone, labeled location, LinkedIn/GitHub/
portfolio links, education, experience, projects, skills, certifications,
achievements, raw extracted text, and lightweight source snippets. Missing fields
are `None` or empty lists. The original text is retained; model preprocessing is
never applied to it. Non-string parser input returns an empty profile. File reading
errors raise `DocumentExtractionError`; the UI and command-line tools report them.

Rules operate on recognized section headings and preserve ambiguous entry text.
Names require an explicit `Name:` label or a short name-shaped header near contact
information; headings, role titles and contact values are rejected. Uncertain names
stay unavailable. Locations require an explicit label. Dates are retained as written;
no dates, experience duration, institutions or employers are inferred.

Skills and aliases live in `src/resume_intelligence/skills.json`. Extend that file
rather than scattering keywords through parser code. Long aliases take precedence:
C, C++, C#, .NET and ASP.NET remain distinct. C/R require uppercase tokens in a skills
context; Go requires that context or the alias Golang. Skills are **text mentions**,
not verified proficiency, endorsements or evidence of suitability for a job.

`src/document_extraction.py` consolidates the existing readers. Line breaks are
preserved for section detection; a small compatibility formatter retains the old
PDF/image/DOCX input shape for classification. TXT decoding replaces invalid bytes.
OCR remains optional and needs the existing pytesseract/pdf2image/Pillow Python
packages plus Tesseract/Poppler system tools. OCR errors are now explicit instead
of silently swallowed. Text PDFs and DOCX/TXT need no OCR installation.

After classification, Streamlit exposes a **Candidate Information** expander for
each readable upload, including the full structured profile and evidence. Existing
classification results remain visible. Uploads use temporary files rather than
writing into the project's output directory.

Run all tests with `python -m unittest discover -s tests -v`. Stage 1 parser unit tests
use synthetic text. Local integration tests additionally use the three PDFs under
`pdf/` when available, without modifying them; those local files are not committed.
OCR routes are tested with mocks, not proof of OCR accuracy on scanned documents.
The existing Stage 0 repeatability test fits temporary in-memory models but does not
replace the saved artifact or evaluation reports.

This is a first deterministic parser, not perfect NLP: unusual layouts, DOCX tables,
columns, unlabeled contact data and entries without clear boundaries can remain
partially parsed. Bullet-only projects keep their descriptions with an unavailable
name. There is no job matching, ranking, database, REST API or recruiter workflow.

## Stage 2: Job intelligence foundation

`Job Description → JobProfile` is available locally through:

```python
from src.job_intelligence import parse_job_description
profile = parse_job_description("Required Skills: Python, Postgres\nPreferred Skills: Docker")
print(profile.to_dict())
```

The dataclass contains title, required/preferred skills, responsibilities, education
and experience requirements, canonical technical keywords, and unchanged raw text.
It reuses Stage 1's skills taxonomy and aliases. Explicit wording overrides section
context; ambiguous qualifications, mixed clauses, alternatives and conflicting
skill statuses are not promoted to mandatory requirements. Unclassified technical
mentions remain in `keywords`, which does not imply a requirement. Education and
experience strings retain their original wording, including preference qualifiers.
Missing fields stay null/empty. This conservative first version expects recognizable
headings or explicit requirement wording and can leave unusual prose unclassified.
There is no UI change; candidate matching, scoring and ranking belong to future stages.

## Stage 3: Single-candidate rule-based matching

```python
from src.matching import match_candidate_to_job
result = match_candidate_to_job(candidate_profile, job_profile)
print(result.to_dict())
```

Matching uses canonical exact skills and only aliases already present in Stage 1's
shared taxonomy. Unlisted skill labels require exact case-insensitive equality;
related skills are not substitutes. Missing skills mean **not found in the profile**,
not proven absent. Matched evidence is copied only from existing candidate evidence.

The result includes matched/missing required and preferred skills, their ratios,
experience/education states, an explanation, evidence, score and assessed-weight
coverage. States are `satisfied`, `not_satisfied`, `unknown`, or `not_required`.
Zero requested skills yield a null ratio, not a perfect or failed match.

The **rule-based match score** is 0–100: required skills 70%, preferred skills 15%,
experience 10%, education 5%. Ratios provide skill component values; satisfied is 1
and not_satisfied is 0. Unknown and unspecified components are excluded and remaining
weights are renormalized. With no assessable components the score is null. Coverage
reports the sum of included original weights; 100 points with low coverage does not
mean every requirement was verified. Scores are not AI predictions or hiring decisions.

Experience checks only one obvious years threshold. An explicit duration at the
start of an experience description (e.g. `3 years Software Engineer`) or closed date
intervals can support it. Overlapping dates are merged; separate duration claims are
never summed. Partial dates use conservative bounds; `Present` is unknown rather than
consulting today's date. Skill-specific tenure, alternatives and optional/ambiguous
requirements remain unknown. Satisfaction establishes duration only, not backend or
other role/domain relevance. Insufficient date evidence stays unknown; a single explicit
duration below the threshold is marked not_satisfied.

Education supports a small set of degree-family equivalents (e.g. B.Tech and
Bachelor's) and exact stated field names. Higher degrees are not automatically
substituted. Missing/nonmatching evidence stays unknown; completion and accreditation
are not verified. Original profiles and ML classification behavior are unchanged.

Example: 3/3 required skills, 1/2 preferred skills, satisfied experience and education
produce `70 + 7.5 + 10 + 5 = 92.5`. If experience and education are unknown, that same
skill overlap scores `(70 + 7.5) / 0.85 = 91.18`, with 85% coverage. No multiple-candidate
ranking, comparison, UI, API, or Stage 4 evidence intelligence is included.
