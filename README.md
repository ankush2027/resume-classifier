# Resume Classification System 🚀

A machine learning system that automatically classifies real-world resumes into exactly **25 job categories**. 

Built to solve a real-world problem: companies receive resumes from multiple platforms (job portals, email, LinkedIn, etc.) in extremely different formats. This system handles them all automatically. It reads standard **PDFs, Word Documents (.docx), Text files (.txt), and even Image-based Scanned Resumes (.png, .jpg) via OCR!**

---

## 🌟 Key Features

- **Multi-Format Extraction:** Automatically extracts text from `.pdf`, `.docx`, and `.txt`.
- **Built-in OCR (Optical Character Recognition):** Can read scanned, photograph-based PDFs, `.jpg`, and `.png` image models via `Tesseract` and `pdf2image`.
- **Hybrid AI Pipeline:** Uses a high-accuracy Machine Learning Model (Logistic Regression / TF-IDF) as the primary pipeline, but auto-falls back to a highly tuned **Domain Keyword System** if ML confidence is low on poorly formatted real-world resumes.
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

**If you are downloading this onto a new Mac/Laptop**, follow these exact steps to make sure Image/OCR scanning works perfectly:

### 1. Install System Requirements (For OCR Image Scanning)
To read scanned PDF images or pictures of resumes, this project relies on Poppler and Tesseract.
If you are on a Mac, run this in your terminal:
```bash
brew install tesseract poppler
```
*(If on Ubuntu: `sudo apt-get install tesseract-ocr poppler-utils`)*

### 2. Set Up Python Environment
```bash
cd resume-classifier
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
*(Note: `requirements.txt` should include `pdfplumber`, `python-docx`, `pytesseract`, `pdf2image`, `pandas`, `scikit-learn`, `pillow`.)*

### 3. Train the Model 🧠
You **must** run this first! It trains the AI on multiple models, compares their accuracy, and saves the perfect one (100% Accuracy on training).
```bash
python3 src/main.py
```

### 4. Batch Classify Real Resumes 📂
**Option A (Folder Mode):** Simply drag and drop all your PDFs, Word files, and Images directly into the `data/input/resumes/` folder!
**Option B (CSV Mode):** If the folder is empty, the code dynamically falls back to the `resumes_to_classify.csv` file automatically.

Then, run:
```bash
python3 src/classify_resumes.py
```
*Results will automatically generate beautifully separated CSVs in the `output/` folder.*

### 5. Check a Single File Interactively 🔍
Need to check just one candidate quickly? Run:
```bash
python3 src/predict.py
```
It will open an interactive prompt. You can **paste a file path** (`/Users/You/resume.pdf` or `.png`) directly, or just copy and paste raw resume text to instantly see a breakdown of the ML prediction and accuracy!

---

## 📌 Troubleshooting
- **`EmptyDataError` when running `main.py`:** Make sure your `data/raw/resume_dataset.csv` file actually has data in it and isn't 0 bytes! (You can type `git restore data/raw/resume_dataset.csv` if you accidentally cleared it).
- **File skipped because of "zlib / corrupted" error:** Sometimes downloaded PDFs are physically corrupted (zero text layer and compressed incorrectly). Open the file in Mac's *Preview* app, select `File -> Export as PDF`, and save a fresh copy. Then the script will read it immediately.
