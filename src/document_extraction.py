"""One shared document reader, consolidated from predict.py and classify_resumes.py.

Layout is preserved for parsing. Legacy wrappers keep the old classifier text shape.
Optional OCR still requires pytesseract, pdf2image, Pillow, Tesseract and Poppler.
"""
from pathlib import Path
import re

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
try:
    import pytesseract
    from pdf2image import convert_from_path
    from PIL import Image
    OCR_SUPPORT = True
except ImportError:
    OCR_SUPPORT = False

SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.txt', '.png', '.jpg', '.jpeg'}


class DocumentExtractionError(ValueError):
    """Unreadable document or unavailable optional extraction dependency."""


def clean_pdf_text(text):
    """Legacy classifier formatting only; never apply to CandidateProfile.raw_text."""
    return ' '.join(line.strip() for line in text.split('\n')
                    if len(line.strip()) >= 3 and not re.match(r'^\d+$', line.strip()))


def classification_text(text, path, *, clean_images=True):
    """Preserve the old PDF/image/DOCX classifier input without losing parser layout."""
    extension = Path(path).suffix.lower()
    if extension == '.pdf' or (clean_images and extension in {'.png', '.jpg', '.jpeg'}):
        return clean_pdf_text(text)
    if extension == '.docx':
        return ' '.join(line for line in text.split('\n') if line.strip())
    return text


def extract_text(path):
    """Return extracted text with line breaks; failures are explicit, never swallowed."""
    path = Path(path)
    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise DocumentExtractionError(f'Unsupported resume format: {extension or "(none)"}')
    try:
        if extension == '.txt':
            return path.read_text(encoding='utf-8-sig', errors='replace')
        if extension == '.docx':
            if not DOCX_SUPPORT:
                raise DocumentExtractionError('DOCX extraction requires python-docx.')
            # Reuse paragraph extraction, retaining line breaks for section boundaries.
            document = Document(path)
            return '\n'.join(paragraph.text for paragraph in document.paragraphs)
        if extension == '.pdf':
            if not PDF_SUPPORT:
                raise DocumentExtractionError('PDF extraction requires pdfplumber.')
            with pdfplumber.open(path) as pdf:
                text = '\n'.join(page.extract_text() or '' for page in pdf.pages)
            if text.strip():
                return text
            if not OCR_SUPPORT:
                raise DocumentExtractionError('PDF has no text layer; OCR dependencies are unavailable.')
            images = convert_from_path(str(path))
            try:
                return '\n'.join(pytesseract.image_to_string(image) for image in images)
            finally:
                for image in images:
                    image.close()
        if not OCR_SUPPORT:
            raise DocumentExtractionError('Image extraction requires the optional OCR dependencies.')
        with Image.open(path) as image:
            return pytesseract.image_to_string(image)
    except DocumentExtractionError:
        raise
    except Exception as error:
        # Third-party readers expose different exception types; preserve the cause.
        raise DocumentExtractionError(f'Could not extract {path.name}: {error}') from error


# Compatibility wrappers retain the existing public reader names without duplicating readers.
def extract_pdf(path):
    return classification_text(extract_text(path), path)


def extract_docx(path):
    return classification_text(extract_text(path), path)


def extract_txt(path):
    return extract_text(path)


def extract_image(path):
    return classification_text(extract_text(path), path)
