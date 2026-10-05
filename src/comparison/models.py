"""Comparison views retain existing ranked candidates and requirement assessments."""
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.evidence.models import RequirementAssessment
from src.ranking import RankedCandidate


@dataclass
class SkillComparison:
    requirement: str
    # None means no assessment record, never an inferred missing skill.
    candidates: Dict[str, Optional[RequirementAssessment]]


@dataclass
class HeadToHead:
    higher_ranked_id: str
    lower_ranked_id: str
    observations: List[str]


@dataclass
class CandidateComparison:
    candidates: List[RankedCandidate]
    required_skills: List[SkillComparison]
    preferred_skills: List[SkillComparison]
    head_to_head: List[HeadToHead]
