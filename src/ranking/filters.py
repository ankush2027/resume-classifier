"""Explicit, explainable filters over existing canonical assessment information."""
from dataclasses import asdict
import math
from typing import List

from src.evidence.rules import MULTIPLIERS
from src.matching.matcher import _skills
from .models import CandidateFilters, ExcludedCandidate, RankedCandidate, RankingResult, requirements_by_skill


def _validate(filters: CandidateFilters) -> None:
    for name, maximum in [('minimum_score', 100), ('minimum_required_skill_coverage', 1)]:
        value = getattr(filters, name)
        if value is not None and (not math.isfinite(value) or not 0 <= value <= maximum):
            raise ValueError(f'{name} must be between 0 and {maximum}.')
    states = {'satisfied', 'not_satisfied', 'unknown', 'not_required'}
    for name in ['experience_status', 'education_status']:
        if getattr(filters, name) is not None and getattr(filters, name) not in states:
            raise ValueError(f'{name} must be an existing matching status.')
    if filters.minimum_evidence_strength is not None and filters.minimum_evidence_strength not in MULTIPLIERS:
        raise ValueError('minimum_evidence_strength must be weak, moderate, or strong.')


def _reasons(candidate: RankedCandidate, filters: CandidateFilters) -> List[str]:
    reasons = []
    if filters.minimum_score is not None:
        score = candidate.ranking_score
        if score is None or not math.isfinite(score):
            reasons.append('Evidence-adjusted score unavailable.')
        elif score < filters.minimum_score:
            reasons.append(f'Evidence-adjusted score {score:g} < minimum {filters.minimum_score:g}.')
    if filters.minimum_required_skill_coverage is not None:
        coverage = candidate.required_skill_coverage
        if coverage is None and candidate.required_skill_names:
            reasons.append('Required-skill coverage unavailable in incomplete assessment.')
        elif coverage is not None and coverage < filters.minimum_required_skill_coverage:
            reasons.append(f'Required-skill coverage {coverage:.0%} < minimum {filters.minimum_required_skill_coverage:.0%}.')
        # A job with no required skills has no coverage constraint to fail.
    for name in ['experience_status', 'education_status']:
        expected = getattr(filters, name)
        actual = getattr(candidate, name)
        if expected is not None and actual != expected:
            reasons.append(f'{name}: {actual}; required {expected}.')
    assessments = requirements_by_skill(candidate.assessment.preferred_requirements + candidate.assessment.required_requirements)
    selected = _skills(filters.required_skills)
    known = _skills(candidate.assessment.match_result.candidate_skills)
    for skill in selected:
        item = assessments.get(skill)
        if item is not None and not item.direct_match:
            reasons.append(f'Required skill {skill}: {item.status}; no direct match.')
        elif item is None and skill not in known:
            reasons.append(f'Required skill {skill}: unavailable in assessed candidate skills.')
    if filters.minimum_evidence_strength is not None:
        # All selected skills must meet the threshold. Otherwise scope it to required
        # job skills, falling back to preferred only for jobs with no required skills.
        scope = selected or candidate.required_skill_names or list(assessments)
        if not scope:
            reasons.append('Skill evidence strength unavailable.')
        for skill in scope:
            item = assessments.get(skill)
            if (item is None or not item.direct_match or
                    MULTIPLIERS.get(item.evidence_strength, 0) < MULTIPLIERS[filters.minimum_evidence_strength]):
                state = (item.evidence_strength if item.direct_match else item.status) if item else 'unknown'
                reasons.append(f'{skill}: {state} evidence; requires direct {filters.minimum_evidence_strength} or stronger evidence.')
    return reasons


def filter_candidates(result: RankingResult, filters: CandidateFilters) -> RankingResult:
    """Reapply filters to the full ranked batch; preserve original ranks and input."""
    _validate(filters)
    included, excluded = [], []
    for candidate in result.ranked_candidates:
        reasons = _reasons(candidate, filters)
        if reasons:
            excluded.append(ExcludedCandidate(candidate, reasons))
        else:
            included.append(candidate)
    return RankingResult(list(result.all_candidates), list(result.ranked_candidates), included, excluded,
                         {'total': len(result.ranked_candidates), 'included': len(included),
                          'excluded': len(excluded), 'filters': asdict(filters)})
