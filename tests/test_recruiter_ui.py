"""Synthetic Stage 6 workflow tests; no personal resume files or external calls."""
import io
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.evidence import assess_candidate
from src.job_intelligence import parse_job_description
from src.predict import hybrid_predict
from src.ranking import rank_candidates
from src.resume_intelligence import parse_resume_file

ROOT = Path(__file__).resolve().parents[1]
JOB = 'Required Skills: Python\nPreferred Skills: Docker\nExperience\n2+ years experience\nEducation\nBachelor degree in Computer Science'
STRONG = 'Name: Asha Rao\nEmail: asha@example.com\nLocation: Pune\nExperience\nSoftware Engineer at Example Labs | Jan 2020 - Dec 2024\nBuilt Python services.\nProjects\nProject: Example\nBuilt Docker containers.\nEducation\nBachelor in Computer Science'
WEAK = 'Name: Rahul Kumar\nSkills\nPython'


def upload(name='resume.txt', text=STRONG):
    value = io.BytesIO(text.encode())
    value.name = name
    return value


@contextmanager
def dashboard(files=None):
    with ExitStack() as stack:
        uploader = stack.enter_context(patch('streamlit.file_uploader', return_value=files or []))
        spies = [stack.enter_context(patch(path, wraps=method)) for path, method in [
            ('src.resume_intelligence.parse_resume_file', parse_resume_file),
            ('src.predict.hybrid_predict', hybrid_predict),
            ('src.evidence.assess_candidate', assess_candidate),
            ('src.ranking.rank_candidates', rank_candidates),
            ('src.job_intelligence.parse_job_description', parse_job_description)]]
        app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=30)
        yield app, uploader, spies


def evaluate(app, job=JOB):
    app.text_area[0].set_value(job).run(timeout=30)
    app.button[0].click().run(timeout=30)
    assert not app.exception
    return app


def notices(app):
    return ' '.join(element.value for element in app.info)


def test_app_load_empty_state():
    with dashboard() as (app, _, spies):
        assert not app.exception
        assert 'Create a job description' in notices(app)
        assert 'Upload one or more resumes' in notices(app)
        assert app.button[0].disabled
        assert not app.dataframe
        assert all(spy.call_count == 0 for spy in spies)


def test_resumes_without_job_do_not_evaluate():
    with dashboard([upload()]) as (app, _, spies):
        assert app.button[0].disabled
        assert not app.dataframe
        assert all(spy.call_count == 0 for spy in spies)


def test_job_without_resumes():
    with dashboard() as (app, _, spies):
        app.text_area[0].set_value(JOB).run()
        assert app.button[0].disabled
        assert not app.dataframe
        assert all(spy.call_count == 0 for spy in spies[:4])
        assert app.session_state['job_profile'].required_skills == ['Python']


def test_title_and_parsed_summary():
    with dashboard([upload()]) as (app, _, _):
        app.text_input[0].set_value('Backend Engineer').run()
        evaluate(app)
        job = app.session_state['job_profile']
        assert job.title == 'Backend Engineer'
        assert job.raw_text == JOB
        assert job.preferred_skills == ['Docker']
        assert any(e.label == 'Parsed job summary' for e in app.expander)


def test_batch_ranking_and_private_table():
    with dashboard([upload('strong.txt'), upload('weak.txt', WEAK)]) as (app, _, spies):
        evaluate(app)
        rows = app.dataframe[-1].value
        assert list(rows['Candidate']) == ['Asha Rao', 'Rahul Kumar']
        assert '@' not in rows.to_string()
        assert ' / 100' in rows.iloc[0]['Evidence-adjusted match']
        assert [spy.call_count for spy in spies[:4]] == [2, 2, 2, 1]
        assert len(app.session_state['evaluation_results']) == 2


def test_filter_selection_and_rerun_reuse_all_expensive_results():
    with dashboard([upload('strong.txt'), upload('weak.txt', WEAK)]) as (app, _, spies):
        evaluate(app, 'Required Skills: Python')
        counts = [spy.call_count for spy in spies]
        select = app.selectbox(key='selected_candidate')
        candidates = list(app.session_state['evaluation_results'])
        select.set_value(candidates[1]).run()
        assert app.session_state['selected_candidate'] == candidates[1]
        assert any(e.value == 'Rahul Kumar' for e in app.subheader)
        app.number_input(key='filter_score').set_value(70).run()
        assert len(app.dataframe[-1].value) == 1
        assert app.session_state['selected_candidate'] == candidates[0]
        assert any(e.value == 'Asha Rao' for e in app.subheader)
        app.run()
        assert not app.exception
        assert [spy.call_count for spy in spies] == counts


@pytest.mark.parametrize('field', ['title', 'description', 'same_name_content', 'filename', 'remove'])
def test_input_changes_clear_results_and_do_not_resurrect_them(field):
    original = upload()
    with dashboard([original]) as (app, uploader, spies):
        evaluate(app)
        counts = [spy.call_count for spy in spies[:4]]
        if field == 'title':
            app.text_input[0].set_value('New job').run()
        elif field == 'description':
            app.text_area[0].set_value('Required Skills: AWS').run()
        else:
            uploader.return_value = {'same_name_content': [upload(text=WEAK)],
                                     'filename': [upload('other.txt')], 'remove': []}[field]
            app.run()
        assert not app.exception
        assert not app.dataframe
        assert 'ranking_result' not in app.session_state
        assert 'evaluation_results' not in app.session_state
        assert 'selected_candidate' not in app.session_state
        assert 'Inputs changed' in notices(app)
        app.text_input[0].set_value('').run()
        app.text_area[0].set_value(JOB).run()
        uploader.return_value = [original]
        app.run()
        assert 'ranking_result' not in app.session_state
        assert [spy.call_count for spy in spies[:4]] == counts


def test_reordering_same_uploads_keeps_results():
    files = [upload('a.txt'), upload('b.txt', WEAK)]
    with dashboard(files) as (app, uploader, spies):
        evaluate(app)
        counts = [spy.call_count for spy in spies]
        uploader.return_value = files[::-1]
        app.run()
        assert app.dataframe
        assert [spy.call_count for spy in spies] == counts


def test_duplicate_names_and_submissions_keep_correct_detail_mapping():
    with dashboard([upload('same.txt'), upload('same.txt', WEAK), upload('same.txt')]) as (app, _, _):
        evaluate(app)
        records = app.session_state['evaluation_results']
        assert len(records) == 3
        for key, record in records.items():
            app.selectbox(key='selected_candidate').set_value(key).run()
            assert any(e.value == record['profile'].candidate_name for e in app.subheader)
        assert not app.exception


def test_no_candidates_pass_filters_and_reset_recovers():
    with dashboard([upload(text=WEAK)]) as (app, _, spies):
        evaluate(app, 'Required Skills: Python')
        calls = spies[2].call_count
        app.number_input(key='filter_score').set_value(100).run()
        assert 'No candidates match' in notices(app)
        assert 'selected_candidate' not in app.session_state
        assert any(e.label == 'Filter exclusions' for e in app.expander)
        app.number_input(key='filter_score').set_value(0).run()
        assert len(app.dataframe[-1].value) == 1
        assert spies[2].call_count == calls


@pytest.mark.parametrize('name,text', [('empty.txt', ''), ('bad.exe', 'unsupported'), ('bad.pdf', 'not a PDF')])
def test_bad_document_does_not_stop_batch(name, text):
    with dashboard([upload(name, text), upload('good.txt')]) as (app, _, _):
        evaluate(app)
        assert len(app.session_state['ranking_result'].ranked_candidates) == 1
        failures = app.session_state['failed_uploads']
        assert len(failures) == 1 and failures[0]['filename'] == name
        assert failures[0]['reason']


def test_all_files_fail_cleanly():
    with dashboard([upload(text='')]) as (app, _, _):
        evaluate(app)
        assert 'No resumes could be evaluated' in notices(app)
        assert not app.dataframe


@pytest.mark.parametrize('stage', ['src.predict.hybrid_predict', 'src.evidence.assess_candidate'])
def test_unexpected_per_file_failure_is_visible_and_batch_continues(stage):
    original = hybrid_predict if stage.endswith('hybrid_predict') else assess_candidate
    calls = []
    def fail_once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError('private internal payload')
        return original(*args, **kwargs)
    with dashboard([upload('first.txt'), upload('second.txt', WEAK)]) as (app, _, _):
        with patch(stage, side_effect=fail_once):
            evaluate(app)
        failures = app.session_state['failed_uploads']
        assert 'Unexpected RuntimeError' in failures[0]['reason']
        assert 'private internal payload' not in failures[0]['reason']
        assert len(app.session_state['ranking_result'].ranked_candidates) == 1


def test_transferable_unknown_and_missing_are_visible_with_original_evidence():
    text = 'Name: Asha Rao\nExperience\nBuilt Azure cloud infrastructure.\nI am learning Python.'
    with dashboard([upload(text=text)]) as (app, _, _):
        evaluate(app, 'Required Skills: AWS, Python, Docker')
        requirement_text = ' '.join(e.value for e in app.markdown)
        assert 'AWS — transferable' in requirement_text
        assert 'Python — unknown' in requirement_text
        assert 'Docker — missing' in requirement_text
        assert 'does not count as direct skill coverage' in notices(app)
        assert any('Built Azure cloud infrastructure.' in e.value for e in app.text)


def test_unknown_name_fallback_and_no_assessable_requirements():
    with dashboard([upload(text='Skills: Python')]) as (app, _, _):
        evaluate(app, 'Join our friendly team.')
        assert app.dataframe[-1].value.iloc[0]['Candidate'] == 'Candidate 1'
        assert app.dataframe[-1].value.iloc[0]['Evidence-adjusted match'] == 'Not assessable'
        assert 'No assessable requirements detected' in notices(app)


def test_missing_model_keeps_app_alive():
    st.cache_resource.clear()
    try:
        with dashboard([upload()]) as (app, _, _):
            with patch('pickle.load', side_effect=FileNotFoundError()):
                evaluate(app)
            assert any('Cannot load' in e.value for e in app.error)
            assert 'ranking_result' not in app.session_state
    finally:
        st.cache_resource.clear()


def test_ranking_failure_does_not_publish_partial_or_stale_results():
    with dashboard([upload()]) as (app, _, _):
        evaluate(app)
        with patch('src.ranking.rank_candidates', side_effect=RuntimeError()):
            app.button[0].click().run()
        assert not app.exception
        assert any('Ranking failed' in e.value for e in app.error)
        assert 'ranking_result' not in app.session_state
        assert 'evaluation_results' not in app.session_state
