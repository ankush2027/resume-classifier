"""Public single-candidate matching API. No ranking or hiring decisions."""
from .models import MatchResult
from .matcher import match_candidate_to_job

__all__ = ['MatchResult', 'match_candidate_to_job']
