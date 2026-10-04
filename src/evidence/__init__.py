"""Evidence-aware view layered on the unchanged Stage 3 matcher."""
from .assessment import assess_candidate
from .models import CandidateAssessment, Evidence, RequirementAssessment

__all__ = ['assess_candidate', 'CandidateAssessment', 'Evidence', 'RequirementAssessment']
