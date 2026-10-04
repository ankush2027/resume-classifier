"""Public Stage 1 interface: raw text or a local document to a candidate profile."""
from .models import CandidateProfile
from .parser import parse_resume
from .extractor import parse_resume_file

__all__ = ['CandidateProfile', 'parse_resume', 'parse_resume_file']
