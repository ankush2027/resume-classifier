"""Rule-based evidence labels are not calibrated confidence or verified claims."""
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Literal, Optional
from src.matching.models import MatchResult

Strength = Literal['weak', 'moderate', 'strong']


@dataclass
class Evidence:
    requirement: str
    concept: str
    source_type: str
    source_text: str
    strength: Strength
    section: str
    relevance: Literal['direct', 'transferable']
    source_path: str
    supports_requirement: bool = True
    reason: str = ''


@dataclass
class RequirementAssessment:
    requirement: str
    status: str
    evidence_strength: Optional[Strength] = None
    direct_match: bool = False
    transferable: bool = False
    evidence: List[Evidence] = field(default_factory=list)
    explanation: str = ''


@dataclass
class ExperienceRelevance:
    requirement: str
    status: Literal['relevant', 'partial', 'unknown']
    source_texts: List[str] = field(default_factory=list)
    explanation: str = ''


@dataclass
class CandidateAssessment:
    candidate_name: Optional[str]
    overall_match_score: Optional[float]
    match_result: MatchResult
    required_requirements: List[RequirementAssessment] = field(default_factory=list)
    preferred_requirements: List[RequirementAssessment] = field(default_factory=list)
    evidence_adjusted_match_score: Optional[float] = None
    score_coverage: float = 0.0
    evidence_strength_summary: Dict[str, int] = field(default_factory=dict)
    experience_relevance: List[ExperienceRelevance] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    concerns: List[str] = field(default_factory=list)
    explanation: str = ''

    def to_dict(self):
        return asdict(self)
