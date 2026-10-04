"""Stage 5 regression and end-to-end behavior, including Streamlit reruns."""
import copy
import io
from pathlib import Path
from unittest.mock import patch

import pytest

from src.evidence import assess_candidate
from src.evidence.models import CandidateAssessment, RequirementAssessment
from src.job_intelligence import parse_job_description
from src.matching.models import MatchResult
from src.ranking import CandidateFilters, filter_candidates, rank_candidates
from src.resume_intelligence import parse_resume
from src.resume_intelligence.models import CandidateProfile


def assessment(name='Candidate', score=80, strength='strong', coverage=.85):
    return CandidateAssessment(
        name, 100, MatchResult(candidate_skills=['Python'], experience_match='unknown',
                               education_match='not_required'),
        required_requirements=[RequirementAssessment('Python', 'evidenced', strength, True)],
        evidence_adjusted_match_score=score, score_coverage=coverage)


def test_basic_ranking():
    result = rank_candidates([assessment('B', 70), assessment('A', 90), assessment('C', 50)])
    assert [c.candidate_name for c in result.ranked_candidates] == ['A', 'B', 'C']
    assert [c.rank for c in result.ranked_candidates] == [1, 2, 3]


def test_deterministic_ties_ignore_upload_order():
    candidates = [assessment('A'), assessment('B'), assessment('C')]
    first = rank_candidates(candidates)
    second = rank_candidates(candidates[::-1])
    assert [(c.candidate_id, c.rank, c.ranking_explanation) for c in first.ranked_candidates] == [
        (c.candidate_id, c.rank, c.ranking_explanation) for c in second.ranked_candidates]
    assert 'stable identifier' in first.ranked_candidates[1].ranking_explanation


def test_required_coverage_tiebreak():
    incomplete = assessment('A')
    incomplete.required_requirements.append(RequirementAssessment('AWS', 'missing'))
    result = rank_candidates([incomplete, assessment('B')])
    assert result.ranked_candidates[0].candidate_name == 'B'
    assert result.ranked_candidates[1].required_skill_coverage == .5


def test_score_coverage_tiebreak():
    assert rank_candidates([assessment('A', coverage=.4), assessment('B', coverage=.95)]).ranked_candidates[0].candidate_name == 'B'


def test_strength_tiebreak():
    assert rank_candidates([assessment('A', strength='weak'), assessment('B')]).ranked_candidates[0].candidate_name == 'B'


def test_primary_score_never_overridden():
    assert rank_candidates([assessment('A', 81, 'weak', .1), assessment('B', 80)]).ranked_candidates[0].candidate_name == 'A'


def test_minimum_score_reason_and_original_rank():
    result = filter_candidates(rank_candidates([assessment('A', 50), assessment('B', 90)]), CandidateFilters(minimum_score=70))
    assert result.filtered_candidates[0].rank == 1
    assert '50 < minimum 70' in result.excluded_candidates[0].reasons[0]


def test_minimum_required_coverage():
    item = assessment()
    item.required_requirements.append(RequirementAssessment('AWS', 'missing'))
    result = filter_candidates(rank_candidates([item]), CandidateFilters(minimum_required_skill_coverage=.75))
    assert '50% < minimum 75%' in result.excluded_candidates[0].reasons[0]


def test_selected_skill_missing():
    item = assessment()
    item.required_requirements.append(RequirementAssessment('AWS', 'missing'))
    result = filter_candidates(rank_candidates([item]), CandidateFilters(required_skills=['AWS']))
    assert 'missing' in result.excluded_candidates[0].reasons[0]


def test_alias_normalization():
    item = assessment()
    item.required_requirements = [RequirementAssessment('PostgreSQL', 'evidenced', 'strong', True)]
    assert filter_candidates(rank_candidates([item]), CandidateFilters(required_skills=['Postgres'])).filtered_candidates


def test_unassessed_skill_presence_from_existing_matcher():
    item = assessment()
    item.match_result.candidate_skills.append('Docker')
    result = rank_candidates([item])
    assert filter_candidates(result, CandidateFilters(required_skills=['Docker'])).filtered_candidates
    assert filter_candidates(result, CandidateFilters(required_skills=['Docker'], minimum_evidence_strength='weak')).excluded_candidates


def test_weak_fails_moderate():
    result = filter_candidates(rank_candidates([assessment(strength='weak')]), CandidateFilters(minimum_evidence_strength='moderate'))
    assert 'weak evidence' in result.excluded_candidates[0].reasons[0]


def test_strong_preferred_does_not_mask_weak_required():
    item = assessment(strength='weak')
    item.preferred_requirements = [RequirementAssessment('Docker', 'evidenced', 'strong', True)]
    assert filter_candidates(rank_candidates([item]), CandidateFilters(minimum_evidence_strength='moderate')).excluded_candidates


def test_selected_strength_scope():
    item = assessment(strength='weak')
    item.preferred_requirements = [RequirementAssessment('Docker', 'evidenced', 'strong', True)]
    assert filter_candidates(rank_candidates([item]), CandidateFilters(required_skills=['Docker'], minimum_evidence_strength='strong')).filtered_candidates


@pytest.mark.parametrize('field', ['experience_status', 'education_status'])
def test_existing_status_filters(field):
    result = rank_candidates([assessment()])
    assert filter_candidates(result, CandidateFilters(**{field: 'satisfied'})).excluded_candidates
    assert filter_candidates(result, CandidateFilters(**{field: getattr(result.ranked_candidates[0], field)})).filtered_candidates


def test_unknown_missing_transferable_remain_distinct():
    item = assessment()
    item.required_requirements += [RequirementAssessment('AWS', 'unknown'),
                                   RequirementAssessment('Docker', 'missing'),
                                   RequirementAssessment('GCP', 'transferable', transferable=True)]
    ranked = rank_candidates([item]).ranked_candidates[0]
    assert ranked.missing_required_skills == ['Docker']
    assert ranked.required_skill_coverage == .25
    assert 'Unknown required skills: AWS' in ranked.ranking_explanation
    assert 'Transferable only (no direct match): GCP' in ranked.ranking_explanation


def test_assessed_unknown_overrides_baseline_keyword():
    item = assessment()
    item.required_requirements[0] = RequirementAssessment('Python', 'unknown')
    assert filter_candidates(rank_candidates([item]), CandidateFilters(required_skills=['Python'])).excluded_candidates


def test_unavailable_score_sorts_after_zero_and_fails_explicit_zero_filter():
    result = rank_candidates([assessment('A', None), assessment('B', 0)])
    assert result.ranked_candidates[0].candidate_name == 'B'
    filtered = filter_candidates(result, CandidateFilters(minimum_score=0))
    assert len(filtered.filtered_candidates) == 1
    assert 'unavailable' in filtered.excluded_candidates[0].reasons[0]


def test_empty_batch():
    result = filter_candidates(rank_candidates([]), CandidateFilters())
    assert result.filter_summary['total'] == 0
    assert result.ranked_candidates == []


def test_no_required_skills():
    item = assessment()
    item.required_requirements = []
    result = filter_candidates(rank_candidates([item]), CandidateFilters(minimum_required_skill_coverage=1))
    assert result.filtered_candidates[0].required_skill_coverage is None
    assert filter_candidates(result, CandidateFilters(minimum_evidence_strength='weak')).excluded_candidates


def test_partial_assessment():
    item = CandidateAssessment(None, None, MatchResult(missing_required_skills=['Python']))
    result = rank_candidates([item])
    assert result.ranked_candidates[0].ranking_score is None
    assert filter_candidates(result, CandidateFilters(minimum_required_skill_coverage=.5)).excluded_candidates


def test_duplicate_candidates_retained_explicitly():
    item = assessment()
    result = rank_candidates([item, copy.deepcopy(item)])
    assert len({c.candidate_id for c in result.ranked_candidates}) == 2
    assert all(c.duplicate_count == 2 for c in result.ranked_candidates)
    assert all('retained separately' in c.ranking_explanation for c in result.ranked_candidates)


def test_profile_identity_private_and_stable_across_jobs():
    profile = CandidateProfile(candidate_name='A', email='a@example.com', raw_text='Python')
    first = rank_candidates([assessment('A')], candidate_profiles=[profile]).ranked_candidates[0]
    second = rank_candidates([assessment('A', 20)], candidate_profiles=[profile]).ranked_candidates[0]
    assert first.candidate_id == second.candidate_id
    assert '@' not in first.candidate_id and 'example' not in first.candidate_id


def test_profile_count_validation():
    with pytest.raises(ValueError):
        rank_candidates([assessment()], candidate_profiles=[])


@pytest.mark.parametrize('values', [dict(minimum_score=-1), dict(minimum_score=float('nan')),
                                    dict(minimum_required_skill_coverage=2), dict(experience_status='partial'),
                                    dict(education_status='missing'), dict(minimum_evidence_strength='excellent')])
def test_invalid_filters(values):
    with pytest.raises(ValueError):
        filter_candidates(rank_candidates([]), CandidateFilters(**values))


def test_input_unchanged_and_refilter_from_full_batch():
    item = assessment()
    before = copy.deepcopy(item.to_dict())
    original = rank_candidates([item])
    excluded = filter_candidates(original, CandidateFilters(minimum_score=90))
    assert original.filtered_candidates
    assert filter_candidates(excluded, CandidateFilters()).filtered_candidates
    assert item.to_dict() == before


def test_explanations_and_multiple_exclusion_reasons():
    result = rank_candidates([assessment(strength='weak')])
    assert result.ranked_candidates[0].ranking_explanation
    filtered = filter_candidates(result, CandidateFilters(minimum_score=90, minimum_evidence_strength='strong'))
    assert len(filtered.excluded_candidates[0].reasons) == 2


def test_keyword_stuffing_end_to_end():
    job = parse_job_description('Required Skills: Python, FastAPI, PostgreSQL, Docker')
    weak = parse_resume('Alice Example\nSkills\n' + 'Python FastAPI PostgreSQL Docker\n' * 20)
    strong = parse_resume('Bob Example\nProfessional Experience\nSoftware Engineer\nBuilt FastAPI services using Python.\nDesigned PostgreSQL schemas.\nContainerized services with Docker.')
    assessments = [assess_candidate(p, job) for p in [weak, strong]]
    result = rank_candidates(assessments, candidate_profiles=[weak, strong])
    assert result.ranked_candidates[0].assessment is assessments[1]
    assert assessments[1].evidence_adjusted_match_score > assessments[0].evidence_adjusted_match_score
    once = assess_candidate(parse_resume('Alice Example\nSkills\nPython FastAPI PostgreSQL Docker'), job)
    assert once.evidence_adjusted_match_score == assessments[0].evidence_adjusted_match_score


def test_transferable_end_to_end():
    job = parse_job_description('Required Skills: AWS')
    candidate = parse_resume('Alex Example\nProfessional Experience\nBuilt cloud infrastructure using Azure.')
    item = assess_candidate(candidate, job)
    result = rank_candidates([item])
    assert item.required_requirements[0].transferable
    assert result.ranked_candidates[0].required_skill_coverage == 0
    assert filter_candidates(result, CandidateFilters(required_skills=['AWS'])).excluded_candidates


def test_repeated_evidence_not_a_tiebreak_reward():
    item = assessment()
    repeated = copy.deepcopy(item)
    repeated.required_requirements *= 10
    assert rank_candidates([item]).ranked_candidates[0].evidence_strength == rank_candidates([repeated]).ranked_candidates[0].evidence_strength


def test_streamlit_batch_filters_persist_and_changed_input_invalidates():
    from streamlit.testing.v1 import AppTest
    first = io.BytesIO(b'Alice Example\nSkills\nPython')
    first.name = 'alice.txt'
    second = io.BytesIO(b'Bob Example\nProfessional Experience\nBuilt Python services.')
    second.name = 'bob.txt'
    with patch('streamlit.file_uploader', return_value=[first, second]), patch(
            'src.evidence.assess_candidate', wraps=assess_candidate) as evaluate:
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=30)
        app.text_area[0].set_value('Required Skills: Python').run(timeout=30)
        app.button[0].click().run(timeout=30)
        assert not app.exception
        assert len(app.dataframe[-1].value) == 2
        assert evaluate.call_count == 2
        app.number_input[0].set_value(70.0).run(timeout=30)
        assert not app.exception
        assert len(app.dataframe[-1].value) == 1
        assert evaluate.call_count == 2
        app.number_input[0].set_value(0.0).run(timeout=30)
        assert len(app.dataframe[-1].value) == 2
        app.text_area[0].set_value('Required Skills: AWS').run(timeout=30)
        assert not app.dataframe
        assert any('Inputs changed' in notice.value for notice in app.info)


def test_partially_missing_requirement_records_do_not_claim_full_coverage():
    item = assessment()
    item.match_result.missing_required_skills = ['AWS']
    result = rank_candidates([item])
    assert result.ranked_candidates[0].required_skill_coverage is None
    assert filter_candidates(result, CandidateFilters(minimum_required_skill_coverage=1)).excluded_candidates
    assert filter_candidates(result, CandidateFilters(minimum_evidence_strength='weak')).excluded_candidates
