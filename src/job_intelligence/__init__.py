"""Public deterministic Job Description → JobProfile API."""
from .models import JobProfile
from .parser import parse_job_description

__all__ = ['JobProfile', 'parse_job_description']
