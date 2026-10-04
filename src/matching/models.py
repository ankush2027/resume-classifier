"""Transparent rule-based results; unknown is distinct from a failed check."""
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Literal, Optional

MatchState = Literal['satisfied', 'not_satisfied', 'unknown', 'not_required']


@dataclass
class MatchResult:
    matched_required_skills: List[str] = field(default_factory=list)
    missing_required_skills: List[str] = field(default_factory=list)
    matched_preferred_skills: List[str] = field(default_factory=list)
    missing_preferred_skills: List[str] = field(default_factory=list)
    candidate_skills: List[str] = field(default_factory=list)
    required_skill_match_ratio: Optional[float] = None
    preferred_skill_match_ratio: Optional[float] = None
    experience_match: MatchState = 'unknown'
    education_match: MatchState = 'unknown'
    overall_match_score: Optional[float] = None
    explanation: str = ''
    matched_skill_evidence: Dict[str, str] = field(default_factory=dict)
    experience_detail: str = ''
    education_detail: str = ''
    score_coverage: float = 0.0

    def to_dict(self):
        return asdict(self)
