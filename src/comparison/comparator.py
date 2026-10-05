"""Describe existing results from one ranked batch, without evaluating or ranking."""
from itertools import combinations
from typing import List, Optional, Sequence

from src.evidence.models import RequirementAssessment
from src.ranking import RankedCandidate, RankingResult
from .models import CandidateComparison, HeadToHead, SkillComparison


def source_strength(requirement: Optional[RequirementAssessment], source: str) -> Optional[str]:
    """Strongest existing supporting direct label for this source, not a score.

    Repetition has no effect. Transferable/negated snippets cannot establish direct
    professional or project evidence, but remain available in the original record.
    """
    if requirement is None or not requirement.direct_match:
        return None
    labels = {e.strength for e in requirement.evidence
              if e.source_type == source and e.relevance == 'direct' and e.supports_requirement}
    return next((label for label in ('strong', 'moderate', 'weak') if label in labels), None)


def _matrix(candidates: List[RankedCandidate], field: str) -> List[SkillComparison]:
    records = {c.candidate_id: {r.requirement: r for r in getattr(c.assessment, field)} for c in candidates}
    # Stage 4 already supplies canonical concepts; comparison does no text matching.
    names = sorted({name for values in records.values() for name in values})
    return [SkillComparison(name, {key: values.get(name) for key, values in records.items()}) for name in names]


def _label(candidate: RankedCandidate) -> str:
    return f'#{candidate.rank} {candidate.candidate_name or "Unnamed candidate"}'


def _metric(label: str, left: Optional[float], right: Optional[float],
            first: str, second: str) -> str:
    if left is None or right is None:
        return f'{label}: {first} = {left if left is not None else "unavailable"}; {second} = {right if right is not None else "unavailable"}. Unavailable is not zero.'
    if left == right:
        return f'{label}: equal ({left:g}).'
    higher = first if left > right else second
    return f'{label}: {first} = {left:g}; {second} = {right:g}. Higher for {higher}.'


def compare_candidates(ranking: RankingResult, candidate_ids: Sequence[str]) -> CandidateComparison:
    """Select 2–4 distinct candidates from one existing batch, in original rank order.

    Filters do not remove candidates from this API: recruiters may compare any
    evaluated submission. No source service, scorer or ranker is invoked.
    """
    if not 2 <= len(candidate_ids) <= 4:
        raise ValueError('Select between 2 and 4 candidates to compare.')
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError('Select distinct candidate identifiers.')
    available = {c.candidate_id: c for c in ranking.ranked_candidates}
    if any(key not in available for key in candidate_ids):
        raise ValueError('Selected candidate is not in the current evaluated batch.')
    selected = sorted((available[key] for key in candidate_ids), key=lambda c: (c.rank, c.candidate_id))
    required = _matrix(selected, 'required_requirements')
    preferred = _matrix(selected, 'preferred_requirements')
    pairs = []
    for left, right in combinations(selected, 2):
        first, second = _label(left), _label(right)
        notes = [f'{first} precedes {second} in the original Stage 5 ranking. Comparison does not change that order.']
        for label, a, b in [
            ('Evidence-adjusted score', left.ranking_score, right.ranking_score),
            ('Direct required-skill coverage (0–1)', left.required_skill_coverage, right.required_skill_coverage),
            ('Assessed weight coverage (0–1)', left.score_coverage, right.score_coverage),
            ('Existing Stage 5 mean evidence strength (0–1)', left.evidence_strength, right.evidence_strength),
        ]:
            notes.append(_metric(label, a, b, first, second))
        for group, rows in [('Required', required), ('Preferred', preferred)]:
            for row in rows:
                a, b = row.candidates[left.candidate_id], row.candidates[right.candidate_id]
                def describe(item):
                    if item is None:
                        return 'unknown / not assessed'
                    return f'{item.status}; direct={item.direct_match}; strength={item.evidence_strength or "unavailable"}; transferable={item.transferable}'
                notes.append(f'{group} {row.requirement}: {first}: {describe(a)}; {second}: {describe(b)}.')
                for source, title in [('experience', 'Professional'), ('project', 'Project')]:
                    strength_a, strength_b = source_strength(a, source), source_strength(b, source)
                    if strength_a or strength_b:
                        notes.append(f'{title} evidence for {row.requirement}: {first}: {strength_a or "no supporting direct snippet"}; '
                                     f'{second}: {strength_b or "no supporting direct snippet"}.')
        pairs.append(HeadToHead(left.candidate_id, right.candidate_id, notes))
    return CandidateComparison(selected, required, preferred, pairs)
