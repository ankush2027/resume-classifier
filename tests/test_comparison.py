"""Stage 7 consumes evaluated records; synthetic UI checks reuse Stage 6 fixtures."""
from copy import deepcopy
from dataclasses import asdict
from unittest.mock import patch

import pytest

from src.comparison import compare_candidates
from src.comparison.comparator import source_strength
from src.evidence import assess_candidate
from src.job_intelligence import parse_job_description
from src.ranking import rank_candidates
from src.resume_intelligence import parse_resume
from test_recruiter_ui import dashboard, evaluate, upload, WEAK


def batch(count=4):
    texts = ['Name: Alpha Example\nExperience\nBuilt Python services on AWS.\nProjects\nBuilt Docker containers.',
             'Name: Beta Example\nSkills\nPython\nExperience\nBuilt Azure cloud infrastructure.',
             'Name: Gamma Example\nProjects\nBuilt Python services.\nExperience\nI am learning AWS.',
             'Name: Delta Example\nSkills\nJava',
             'Name: Epsilon Example\nSkills\nSQL']
    job = parse_job_description('Required Skills: Python\nPreferred Skills: AWS, Docker')
    assessments = [assess_candidate(parse_resume(text), job) for text in texts[:count]]
    return rank_candidates(assessments)


def ids(result):
    return [c.candidate_id for c in result.ranked_candidates]


@pytest.mark.parametrize('count', [2, 3, 4])
def test_compare_supported_counts(count):
    ranking = batch(count)
    result = compare_candidates(ranking, ids(ranking))
    assert len(result.candidates) == count
    assert len(result.head_to_head) == count * (count - 1) // 2


@pytest.mark.parametrize('count', [0, 1, 5])
def test_reject_invalid_counts(count):
    ranking = batch(5)
    with pytest.raises(ValueError, match='between 2 and 4'):
        compare_candidates(ranking, ids(ranking)[:count])


def test_reject_duplicate_selection():
    ranking = batch()
    with pytest.raises(ValueError, match='distinct'):
        compare_candidates(ranking, [ids(ranking)[0]] * 2)


def test_reject_stale_identifier():
    ranking = batch()
    with pytest.raises(ValueError, match='current evaluated batch'):
        compare_candidates(ranking, [ids(ranking)[0], 'old-candidate'])


def test_existing_scores_objects_and_inputs_preserved_without_services():
    ranking = batch()
    before = deepcopy(asdict(ranking))
    with patch('src.ranking.rank_candidates', side_effect=AssertionError('must not rank')), patch(
            'src.evidence.assess_candidate', side_effect=AssertionError('must not evaluate')):
        result = compare_candidates(ranking, ids(ranking))
    for original, compared in zip(ranking.ranked_candidates, result.candidates):
        assert compared is original
        assert compared.ranking_score == original.assessment.evidence_adjusted_match_score
    assert asdict(ranking) == before


def test_required_and_preferred_states_preserved():
    ranking = batch()
    result = compare_candidates(ranking, ids(ranking))
    assert [r.requirement for r in result.required_skills] == ['Python']
    assert [r.requirement for r in result.preferred_skills] == ['AWS', 'Docker']
    for row in result.required_skills + result.preferred_skills:
        for candidate in result.candidates:
            original = next(r for r in candidate.assessment.required_requirements + candidate.assessment.preferred_requirements
                            if r.requirement == row.requirement)
            assert row.candidates[candidate.candidate_id] is original


def test_direct_transferable_unknown_missing_distinct():
    ranking = batch()
    result = compare_candidates(ranking, ids(ranking))
    aws = next(r for r in result.preferred_skills if r.requirement == 'AWS')
    by_name = {c.candidate_name: aws.candidates[c.candidate_id] for c in result.candidates}
    assert by_name['Alpha Example'].direct_match
    assert by_name['Beta Example'].transferable and not by_name['Beta Example'].direct_match
    assert by_name['Gamma Example'].status == 'unknown'
    assert by_name['Delta Example'].status == 'missing'


def test_absent_record_is_unassessed_not_missing():
    ranking = batch(2)
    ranking.ranked_candidates[1].assessment.required_requirements = []
    result = compare_candidates(ranking, ids(ranking))
    assert result.required_skills[0].candidates[ids(ranking)[1]] is None
    assert any('unknown / not assessed' in note for note in result.head_to_head[0].observations)


def test_evidence_strength_and_source_preserved():
    ranking = batch(3)
    result = compare_candidates(ranking, ids(ranking))
    row = result.required_skills[0]
    by_name = {c.candidate_name: row.candidates[c.candidate_id] for c in result.candidates}
    assert by_name['Alpha Example'].evidence_strength == 'strong'
    assert by_name['Beta Example'].evidence_strength == 'weak'
    assert by_name['Gamma Example'].evidence_strength == 'moderate'
    assert source_strength(by_name['Alpha Example'], 'experience') == 'strong'
    assert source_strength(by_name['Gamma Example'], 'project') == 'moderate'
    assert source_strength(by_name['Gamma Example'], 'experience') is None


def test_transferable_and_unsupported_not_professional_direct_support():
    ranking = batch(3)
    result = compare_candidates(ranking, ids(ranking))
    aws = next(r for r in result.preferred_skills if r.requirement == 'AWS')
    for candidate in result.candidates:
        if candidate.candidate_name != 'Alpha Example':
            assert source_strength(aws.candidates[candidate.candidate_id], 'experience') is None


def test_deterministic_input_order_does_not_change_result():
    ranking = batch()
    assert asdict(compare_candidates(ranking, ids(ranking))) == asdict(compare_candidates(ranking, ids(ranking)[::-1]))


def test_original_ranks_not_renumbered_for_subset():
    ranking = batch()
    result = compare_candidates(ranking, ids(ranking)[1::2])
    assert [c.rank for c in result.candidates] == [2, 4]


def test_unavailable_not_converted_to_zero():
    ranking = batch(2)
    ranking.ranked_candidates[1].assessment.evidence_adjusted_match_score = None
    result = compare_candidates(ranking, ids(ranking))
    assert result.candidates[1].ranking_score is None
    assert any('Unavailable is not zero' in note for note in result.head_to_head[0].observations)


def test_head_to_head_reports_independent_dimensions_not_new_ranking():
    ranking = batch(2)
    ranking.ranked_candidates[0].assessment.score_coverage = .4
    ranking.ranked_candidates[1].assessment.score_coverage = .95
    result = compare_candidates(ranking, ids(ranking))
    assert result.candidates[0] is ranking.ranked_candidates[0]
    assert any('Assessed weight coverage' in note and 'Higher for #2' in note for note in result.head_to_head[0].observations)


def test_repeated_evidence_records_add_no_comparison_advantage():
    ranking = batch(2)
    before = asdict(compare_candidates(ranking, ids(ranking)).head_to_head[0])
    for requirement in ranking.ranked_candidates[0].assessment.required_requirements:
        requirement.evidence *= 20
    assert asdict(compare_candidates(ranking, ids(ranking)).head_to_head[0]) == before


def test_keyword_stuffing_cannot_overtake_work_evidence():
    job = parse_job_description('Required Skills: Python, FastAPI')
    strong = assess_candidate(parse_resume('Name: Work Example\nExperience\nBuilt Python services using FastAPI.'), job)
    stuffed = assess_candidate(parse_resume('Name: List Example\nSkills\n' + 'Python FastAPI\n' * 20), job)
    ranking = rank_candidates([stuffed, strong])
    result = compare_candidates(ranking, ids(ranking)[::-1])
    assert result.candidates[0].candidate_name == 'Work Example'
    assert result.candidates[0].ranking_score > result.candidates[1].ranking_score


@pytest.mark.parametrize('count', [2, 4])
def test_streamlit_comparison_and_clear_without_reevaluation(count):
    files = [upload(f'{i}.txt', f'Name: Person Example\nSkills\nPython\nProjects\nBuilt Python services.\n{i}') for i in range(count)]
    with dashboard(files) as (app, _, spies):
        evaluate(app, 'Required Skills: Python')
        keys = ids(app.session_state['ranking_result'])
        before = [spy.call_count for spy in spies]
        app.multiselect(key='comparison_selection').set_value(keys[::-1]).run()
        assert not app.exception
        assert len(app.table) == 3  # overview, required, evidence; no preferred requirements
        assert len(app.table[0].value.columns) == count
        assert list(app.table[0].value.loc['Original rank']) == [str(i) for i in range(1, count + 1)]
        app.selectbox(key='selected_candidate').set_value(keys[-1]).run()
        assert app.session_state['comparison_selection'] == keys[::-1]
        app.number_input(key='filter_score').set_value(100).run()
        assert len(app.table[0].value.columns) == count  # comparison spans entire batch
        app.button(key='comparison_clear').click().run()
        assert app.session_state['comparison_selection'] == []
        assert not app.table
        assert [spy.call_count for spy in spies] == before


@pytest.mark.parametrize('change', ['job', 'upload', 'reevaluate'])
def test_comparison_invalidated_with_batch(change):
    with dashboard([upload('a.txt'), upload('b.txt', WEAK)]) as (app, uploader, _):
        evaluate(app)
        app.multiselect(key='comparison_selection').set_value(ids(app.session_state['ranking_result'])).run()
        assert app.table
        if change == 'job':
            app.text_area[0].set_value('Required Skills: AWS').run()
        elif change == 'upload':
            uploader.return_value = [upload('a.txt', WEAK), upload('b.txt', WEAK)]
            app.run()
        else:
            app.button[0].click().run()
        assert not app.exception
        assert not app.table
        assert 'comparison_selection' not in app.session_state or app.session_state['comparison_selection'] == []


def test_comparison_not_available_before_evaluation():
    with dashboard() as (app, _, _):
        assert not any(widget.key == 'comparison_selection' for widget in app.multiselect)


def test_one_selection_message_and_four_selection_limit():
    with dashboard([upload('a.txt'), upload('b.txt', WEAK)]) as (app, _, _):
        evaluate(app)
        widget = app.multiselect(key='comparison_selection')
        assert widget.proto.max_selections == 4
        widget.set_value(ids(app.session_state['ranking_result'])[:1]).run()
        assert not app.table
        assert any('Select at least 2' in element.value for element in app.info)
