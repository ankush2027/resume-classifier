import streamlit as st
import pandas as pd
import pickle
import os
import sys

# Configure Streamlit page for a modern look
st.set_page_config(
    page_title="Resume Classifier AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import our backend logic safely
# Since we wrapped the execution in if __name__ == '__main__': this is safe.
from src.predict import hybrid_predict, extract_pdf, extract_docx, clean_pdf_text, OCR_SUPPORT
try:
    from PIL import Image
except ImportError:
    pass
try:
    import pytesseract
except ImportError:
    pass

# Custom CSS for styling
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .main-header {
        font-size: 40px;
        font-weight: 700;
        margin-bottom: 0px;
        background: -webkit-linear-gradient(45deg, #4facfe, #00f2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 18px;
        color: #a0aab2;
        margin-bottom: 30px;
    }
    .card {
        background-color: #1e2229;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #2d333b;
        margin-bottom: 20px;
    }
    .prediction-title {
        color: #4facfe;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Load the model and vectorizer
@st.cache_resource
def load_classification_model():
    model_path = os.path.join("models", "model.pkl")
    if not os.path.exists(model_path):
        return None, None
    with open(model_path, "rb") as f:
        m, v = pickle.load(f)
    return m, v

model, tfidf = load_classification_model()

# Header
st.markdown('<div class="main-header">Resume Classifier AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload resumes (PDF, DOCX, TXT) and instantly predict their job category using Hybrid Machine Learning.</div>', unsafe_allow_html=True)

if model is None or tfidf is None:
    st.error("Model not found! Please ensure you have trained the model using `python src/main.py` and that `models/model.pkl` exists.")
    st.stop()

# Batch Processing
st.markdown("### Upload Resumes for Classification")
uploaded_files = st.file_uploader("Select PDF, DOCX, or TXT files", type=["pdf", "docx", "txt", "png", "jpg", "jpeg"], accept_multiple_files=True)

if st.button("Classify Resumes"):
    if not uploaded_files:
        st.warning("Please upload at least one resume file.")
    else:
        results = []
        skipped = []

        my_bar = st.progress(0)
        
        for i, uploaded_file in enumerate(uploaded_files):
            # Save uploaded file temporarily to use file paths like the original system
            temp_path = os.path.join("output", uploaded_file.name)
            os.makedirs("output", exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            ext = os.path.splitext(uploaded_file.name)[1].lower()
            raw_text = ""
            
            # File Extraction Logic
            if ext == ".pdf":
                raw_text = extract_pdf(temp_path)
            elif ext == ".docx":
                raw_text = extract_docx(temp_path)
            elif ext == ".txt":
                with open(temp_path, "r", encoding="utf-8", errors="ignore") as f:
                    raw_text = f.read()
            elif ext in [".png", ".jpg", ".jpeg"]:
                if OCR_SUPPORT:
                    try:
                        img = Image.open(temp_path)
                        raw_text = pytesseract.image_to_string(img)
                    except Exception:
                        pass
                
            # Image-based PDF fallback
            cleaned_raw = clean_pdf_text(raw_text)
            if not cleaned_raw.strip():
                skipped.append(uploaded_file.name)
            else:
                # Prediction
                prediction, confidence, top3, method = hybrid_predict(raw_text, model, tfidf)
                results.append({
                    "Filename": uploaded_file.name,
                    "Predicted Category": prediction,
                    "Confidence": f"{confidence:.1f}%",
                    "Method": method
                })

            # Cleanup Temp File
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            # Update Progress Bar
            my_bar.progress((i + 1) / len(uploaded_files))
            
        st.success("Classification Complete!")
        
        # Display Skipped files
        if skipped:
            st.warning(f"⚠️ Skipped {len(skipped)} files due to unreadable content or image-based PDFs without OCR fallback.")
            with st.expander("Show skipped files"):
                for s in skipped:
                    st.write(f"- {s}")

        # Display Results
        if results:
            df_results = pd.DataFrame(results)
            
            st.markdown("### Classification Results")
            st.dataframe(df_results, use_container_width=True)
            
            # Grouping visually
            st.markdown("### Job Category Breakdown")
            category_counts = df_results["Predicted Category"].value_counts().reset_index()
            category_counts.columns = ["Category", "Count"]
            
            cols = st.columns(3)
            for idx, row in category_counts.iterrows():
                with cols[idx % 3]:
                    st.metric(label=row["Category"], value=f"{row['Count']} Resumes")

            st.markdown("### Download Results")
            csv = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV Report",
                data=csv,
                file_name='resume_classification_results.csv',
                mime='text/csv',
            )
