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
from src.predict import hybrid_predict
from src.document_extraction import DocumentExtractionError, classification_text
from src.resume_intelligence import parse_resume_file
from pathlib import Path
from tempfile import TemporaryDirectory

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

# Load the complete raw-text pipeline
@st.cache_resource
def load_classification_model():
    model_path = os.path.join("models", "model.pkl")
    if not os.path.exists(model_path):
        return None
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

model = load_classification_model()

# Header
st.markdown('<div class="main-header">Resume Classifier AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload resumes (PDF, DOCX, TXT) and instantly predict their job category using Hybrid Machine Learning.</div>', unsafe_allow_html=True)

if model is None:
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
        profiles = []

        my_bar = st.progress(0)
        
        for i, uploaded_file in enumerate(uploaded_files):
            # Isolate uploads so filenames cannot overwrite existing project files.
            with TemporaryDirectory(prefix="resume-upload-") as directory:
                temp_path = Path(directory) / Path(uploaded_file.name).name
                temp_path.write_bytes(uploaded_file.getbuffer())
                try:
                    profile = parse_resume_file(temp_path)
                except DocumentExtractionError as error:
                    skipped.append(f"{uploaded_file.name}: {error}")
                    my_bar.progress((i + 1) / len(uploaded_files))
                    continue

            raw_text = classification_text(profile.raw_text, uploaded_file.name, clean_images=False)
            if not raw_text.strip():
                skipped.append(uploaded_file.name)
            else:
                prediction, confidence, top3, method = hybrid_predict(raw_text, model)
                results.append({
                    "Filename": uploaded_file.name,
                    "Predicted Category": prediction,
                    "Confidence": f"{confidence:.1f}%" if confidence is not None else "N/A",
                    "Method": method
                })
                profiles.append((uploaded_file.name, profile))

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

            for filename, profile in profiles:
                with st.expander(f"Candidate Information — {filename}"):
                    st.write({"Name": profile.candidate_name, "Email": profile.email,
                              "Phone": profile.phone, "Location": profile.location})
                    st.write("Skills", profile.skills)
                    # JSON keeps nullable fields, source text and evidence inspectable.
                    st.json(profile.to_dict())

            
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
