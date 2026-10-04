"""Views over existing assessments; ranking does not replace evaluation."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.evidence.models import CandidateAssessment, RequirementAssessment, Strength
from src.evidence.rules import MULTIPLIERS
from src.matching.models import MatchState
from src.matching.matcher import _skills


def requirements_by_skill(requirements: List[RequirementAssessment]) -> Dict[str, RequirementAssessment]:
    # Stage 4 already emits unique canonical requirements. Preserve one per concept.
    return {skill: item for item in requirements for skill in _skills([item.requirement])}


@dataclass
class RankedCandidate:
    candidate_id: str
    candidate_name: Optional[str]
    assessment: CandidateAssessment
    rank: int = 0
    duplicate_count: int = 1
    ranking_explanation: str = ''

    @property
    def ranking_score(self) -> Optional[float]:
        return self.assessment.evidence_adjusted_match_score

    @property
    def score_coverage(self) -> float:
        return self.assessment.score_coverage

    @property
    def required_skill_names(self) -> List[str]:
        baseline = self.assessment.match_result
        return _skills([r.requirement for r in self.assessment.required_requirements] +
                       baseline.matched_required_skills + baseline.missing_required_skills)

    @property
    def required_skill_coverage(self) -> Optional[float]:
        requirements = requirements_by_skill(self.assessment.required_requirements)
        if not requirements or set(self.required_skill_names) - requirements.keys():
            return None  # No requirements (or an incomplete assessment), not 100%.
        return sum(item.direct_match for item in requirements.values()) / len(requirements)

    @property
    def evidence_strength(self) -> Optional[float]:
        requirements = requirements_by_skill(
            self.assessment.preferred_requirements + self.assessment.required_requirements)
        if not requirements:
            return None
        # Mean per unique requirement, never a sum over evidence records.
        return sum(MULTIPLIERS.get(item.evidence_strength, 0) if item.direct_match else 0
                   for item in requirements.values()) / len(requirements)

    @property
    def evidence_strength_summary(self) -> Dict[str, int]:
        return self.assessment.evidence_strength_summary

    @property
    def matched_required_skills(self) -> List[str]:
        return [r.requirement for r in self.assessment.required_requirements if r.direct_match]

    @property
    def missing_required_skills(self) -> List[str]:
        return [r.requirement for r in self.assessment.required_requirements if r.status == 'missing']

    @property
    def matched_preferred_skills(self) -> List[str]:
        return [r.requirement for r in self.assessment.preferred_requirements if r.direct_match]

    @property
    def missing_preferred_skills(self) -> List[str]:
        return [r.requirement for r in self.assessment.preferred_requirements if r.status == 'missing']

    @property
    def experience_status(self) -> MatchState:
        return self.assessment.match_result.experience_match

    @property
    def education_status(self) -> MatchState:
        return self.assessment.match_result.education_match

    @property
    def strengths(self) -> List[str]:
        return self.assessment.strengths

    @property
    def concerns(self) -> List[str]:
        return self.assessment.concerns


@dataclass
class CandidateFilters:
    minimum_score: Optional[float] = None
    minimum_required_skill_coverage: Optional[float] = None
    required_skills: List[str] = field(default_factory=list)
    experience_status: Optional[MatchState] = None
    education_status: Optional[MatchState] = None
    minimum_evidence_strength: Optional[Strength] = None


@dataclass
class ExcludedCandidate:
    candidate: RankedCandidate
    reasons: List[str]


@dataclass
class RankingResult:
    all_candidates: List[RankedCandidate] = field(default_factory=list)
    ranked_candidates: List[RankedCandidate] = field(default_factory=list)
    filtered_candidates: List[RankedCandidate] = field(default_factory=list)
    excluded_candidates: List[ExcludedCandidate] = field(default_factory=list)
    filter_summary: Dict[str, object] = field(default_factory=dict)
