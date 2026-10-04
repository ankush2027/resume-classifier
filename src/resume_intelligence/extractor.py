"""The intelligence layer delegates all document reading to the shared extractor."""
from src.document_extraction import extract_text
from .parser import parse_resume
from .models import CandidateProfile


def parse_resume_file(path) -> CandidateProfile:
    return parse_resume(extract_text(path))
