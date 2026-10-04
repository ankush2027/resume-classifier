"""Lexicographic ordering of already evaluated candidates for the same job."""
import hashlib
import json
import math
from collections import Counter
from typing import Optional, Sequence

from src.evidence.models import CandidateAssessment
from src.resume_intelligence.models import CandidateProfile
from .models import RankedCandidate, RankingResult


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _number(value: Optional[float]) -> float:
    return value if value is not None and math.isfinite(value) else -1.0


def _key(candidate: RankedCandidate) -> tuple:
    return (-_number(candidate.ranking_score), -_number(candidate.required_skill_coverage),
            -_number(candidate.score_coverage), -_number(candidate.evidence_strength),
            candidate.candidate_id)


def _explain(candidate: RankedCandidate, previous: Optional[RankedCandidate]) -> str:
    score = candidate.ranking_score
    parts = [f'Ranked #{candidate.rank}; evidence-adjusted score: {score if _number(score) >= 0 else "unavailable"}.',
             f'Assessed weight coverage: {candidate.score_coverage:.0%}.']
    coverage = candidate.required_skill_coverage
    parts.append(f'Direct required-skill coverage: {coverage:.0%}.' if coverage is not None
                 else 'Required-skill coverage: not applicable or unavailable; see assessment.')
    for label, predicate in [
        ('Missing required skills', lambda r: r.status == 'missing'),
        ('Unknown required skills', lambda r: r.status == 'unknown'),
        ('Transferable only (no direct match)', lambda r: r.transferable and not r.direct_match),
        ('Weak required evidence', lambda r: r.direct_match and r.evidence_strength == 'weak'),
    ]:
        names = [r.requirement for r in candidate.assessment.required_requirements if predicate(r)]
        if names:
            parts.append(label + ': ' + ', '.join(names) + '.')
    parts.append(f'Experience: {candidate.experience_status}; education: {candidate.education_status}.')
    if candidate.score_coverage < 1:
        parts.append('Coverage excludes unknown or unspecified components; it is not evidence of inability.')
    if previous:
        labels = ['evidence-adjusted score', 'direct required-skill coverage',
                  'assessed weight coverage', 'mean requirement evidence strength', 'stable identifier']
        deciding = next(label for label, left, right in zip(labels, _key(previous), _key(candidate)) if left != right)
        parts.append(f'Ordered after #{previous.rank} by {deciding}.')
    else:
        parts.append('First in this batch under the documented score, coverage, strength, and identifier ordering.')
    if candidate.duplicate_count > 1:
        parts.append(f'{candidate.duplicate_count} submissions share this identity; retained separately, not merged.')
    return ' '.join(parts)


def rank_candidates(assessments: Sequence[CandidateAssessment], *,
                    candidate_profiles: Optional[Sequence[CandidateProfile]] = None) -> RankingResult:
    """Profiles are optional identity inputs only. All assessments must concern one job.

    IDs hash profile content (no contact details in the returned identifier). Without
    profiles they hash the assessment and are specific to that evaluation. Identical
    submissions receive occurrence suffixes and remain interchangeable, not merged.
    """
    if candidate_profiles is not None and len(candidate_profiles) != len(assessments):
        raise ValueError('Provide one candidate profile per assessment.')
    records = []
    for index, assessment in enumerate(assessments):
        identity = assessment.to_dict()
        if candidate_profiles is not None:
            identity = candidate_profiles[index].to_dict()
            identity['email'] = (identity.get('email') or '').strip().casefold()
        records.append((_digest(identity), _digest(assessment.to_dict()), assessment))
    counts = Counter(identity for identity, _, _ in records)
    occurrences = Counter()
    candidates = []
    for identity, _, assessment in sorted(records, key=lambda row: row[:2]):
        occurrences[identity] += 1
        identifier = 'candidate-' + identity
        if counts[identity] > 1:
            identifier += f'-{occurrences[identity]}'
        candidates.append(RankedCandidate(identifier, assessment.candidate_name, assessment,
                                          duplicate_count=counts[identity]))
    ranked = sorted(candidates, key=_key)
    for index, candidate in enumerate(ranked):
        candidate.rank = index + 1
        candidate.ranking_explanation = _explain(candidate, ranked[index - 1] if index else None)
    return RankingResult(candidates, ranked, list(ranked), [],
                         {'total': len(ranked), 'included': len(ranked), 'excluded': 0, 'filters': {}})
