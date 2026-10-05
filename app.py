"""Local recruiter workflow orchestrating the existing Stage 0–5 services."""
import hashlib
import pickle
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import streamlit as st

from src.document_extraction import DocumentExtractionError, classification_text
from src.evidence import assess_candidate
from src.job_intelligence import parse_job_description
from src.predict import hybrid_predict
from src.ranking import filter_candidates, rank_candidates
from src.resume_intelligence import parse_resume_file
from src.recruiter_ui import (candidate_label, percentage, score, show_candidate,
                              show_filters, show_job, show_comparison)

st.set_page_config(page_title='Resume Intelligence & Candidate Evaluation', page_icon='📄', layout='wide')
st.title('Resume Intelligence & Candidate Evaluation')
st.caption('Evaluate resume evidence against a job, filter the ranked results, and inspect the reasons. '
           'Decision support for recruiter review; claims are not independently verified.')


@st.cache_resource
def load_classification_model():
    path = Path(__file__).resolve().parent / 'models' / 'model.pkl'
    with path.open('rb') as stream:
        return pickle.load(stream)


st.header('1. Job Setup')
job_title = st.text_input('Job Title (optional)', key='input_job_title')
job_description = st.text_area('Job Description', key='input_job_description')
job_summary = st.container()
st.header('2. Resumes')
uploaded_files = st.file_uploader('Select PDF, DOCX, TXT, or supported images',
                                type=['pdf', 'docx', 'txt', 'png', 'jpg', 'jpeg'],
                                accept_multiple_files=True, key='input_resumes') or []
st.write(f'Selected resumes: {len(uploaded_files)}')
metadata = [{'name': upload.name, 'size': len(upload.getbuffer()),
             'digest': hashlib.sha256(upload.getbuffer()).hexdigest()} for upload in uploaded_files]
with st.expander('Selected files', expanded=bool(uploaded_files)):
    for item in metadata:
        st.text(f'{item["name"]} · {item["size"]:,} bytes')

# Content hashes detect same-name replacements. Sorting ignores upload order while
# retaining duplicate multiplicity. No resume text or contact information is logged.
signature = (job_title, job_description, tuple(sorted((m['name'], m['digest']) for m in metadata)))
if st.session_state.get('input_signature') != signature:
    had_results = 'ranking_result' in st.session_state
    for key in list(st.session_state):
        if key.startswith('filter_') or key in {
                'evaluation_results', 'ranking_result', 'failed_uploads',
                'selected_candidate', 'evaluation_signature', 'comparison_selection'}:
            del st.session_state[key]
    st.session_state['input_signature'] = signature
    st.session_state['uploaded_resume_metadata'] = metadata
    parsed_job = parse_job_description(job_description) if job_description.strip() else None
    if parsed_job is not None and job_title.strip():
        parsed_job = replace(parsed_job, title=job_title.strip())
    st.session_state['job_profile'] = parsed_job
    if had_results:
        st.info('Inputs changed. Evaluate Candidates again to see current results.')
job = st.session_state.get('job_profile')
with job_summary:
    if job is None:
        st.info('Create a job description to begin candidate evaluation.')
    else:
        show_job(job)
if not uploaded_files:
    st.info('Upload one or more resumes to evaluate candidates.')

st.header('3. Evaluate')
if st.button('Evaluate Candidates', type='primary', disabled=job is None or not uploaded_files):
    # A failed retry must not leave old results visible.
    for key in ['evaluation_results', 'ranking_result', 'failed_uploads',
                'selected_candidate', 'evaluation_signature', 'comparison_selection']:
        st.session_state.pop(key, None)
    try:
        model = load_classification_model()
    except Exception as error:
        st.error(f'Cannot load the saved classification model ({type(error).__name__}). '
                 'Check models/model.pkl and the pinned dependencies. No candidates were evaluated.')
    else:
        records, failures = [], []
        progress = st.progress(0, text='Evaluating candidates…')
        for index, upload in enumerate(uploaded_files):
            phase = 'document extraction'
            progress.progress(index / len(uploaded_files), text=f'Evaluating {index + 1}/{len(uploaded_files)}: {upload.name}')
            try:
                with TemporaryDirectory(prefix='resume-upload-') as directory:
                    path = Path(directory) / Path(upload.name).name
                    path.write_bytes(upload.getbuffer())
                    profile = parse_resume_file(path)
                raw_text = classification_text(profile.raw_text, upload.name, clean_images=False)
                if not raw_text.strip():
                    raise DocumentExtractionError('No readable resume text found.')
                phase = 'classification'
                prediction, confidence, _, method = hybrid_predict(raw_text, model)
                classification = {'Filename': upload.name, 'Predicted Category': prediction,
                                  'Confidence': f'{confidence:.1f}%' if confidence is not None else 'N/A',
                                  'Method': method}
                phase = 'evidence assessment'
                assessment = assess_candidate(profile, job)
                records.append({'filename': upload.name, 'profile': profile,
                                'classification': classification, 'assessment': assessment})
            except DocumentExtractionError as error:
                failures.append({'filename': upload.name, 'reason': str(error)})
            except Exception as error:
                # Batch boundary: isolate unexpected failures without hiding their phase
                # or exposing exception payloads that could contain resume/contact text.
                failures.append({'filename': upload.name,
                                 'reason': f'Unexpected {type(error).__name__} during {phase}. '
                                           'Retry this file or investigate the local component.'})
            progress.progress((index + 1) / len(uploaded_files), text=f'Processed {index + 1}/{len(uploaded_files)}')
        st.session_state['failed_uploads'] = failures
        try:
            ranking = rank_candidates([r['assessment'] for r in records],
                                      candidate_profiles=[r['profile'] for r in records])
        except Exception as error:
            st.error(f'Ranking failed ({type(error).__name__}). Results were not published; try evaluating again.')
        else:
            # Match by the exact retained assessment, not name or upload position:
            # names and even identical submissions can be duplicated.
            by_assessment = {id(record['assessment']): record for record in records}
            st.session_state['evaluation_results'] = {
                candidate.candidate_id: by_assessment[id(candidate.assessment)]
                for candidate in ranking.ranked_candidates}
            st.session_state['ranking_result'] = ranking
            st.session_state['evaluation_signature'] = signature

failures = st.session_state.get('failed_uploads', [])
if failures:
    with st.expander(f'Failed files ({len(failures)})', expanded=True):
        for failure in failures:
            st.text(f'{failure["filename"]}: {failure["reason"]}')

ranking = st.session_state.get('ranking_result')
if ranking is not None:
    records = st.session_state['evaluation_results']
    st.header('4. Candidate Ranking')
    st.write(f'{len(ranking.ranked_candidates)} evaluated successfully · {len(failures)} failed')
    if not ranking.ranked_candidates:
        st.info('No resumes could be evaluated. Review failed files and try again.')
    else:
        # Preserve the earlier classification view and existing CSV download.
        with st.expander('Classification results'):
            classification = pd.DataFrame([record['classification'] for record in records.values()])
            st.dataframe(classification, hide_index=True, use_container_width=True)
            st.write('Job category breakdown', classification['Predicted Category'].value_counts().to_dict())
            st.download_button('Download classification CSV', classification.to_csv(index=False).encode('utf-8'),
                               file_name='resume_classification_results.csv', mime='text/csv')
        filtered = filter_candidates(ranking, show_filters(ranking))
        st.write(f'Showing {len(filtered.filtered_candidates)} of {len(ranking.ranked_candidates)} candidates '
                 f'· {len(filtered.excluded_candidates)} excluded by filters')
        with st.expander('Filter exclusions'):
            for excluded in filtered.excluded_candidates:
                st.write(candidate_label(excluded.candidate), excluded.reasons)
        if not filtered.filtered_candidates:
            st.session_state.pop('selected_candidate', None)
            st.info('No candidates match the current filters. Try relaxing one or more filters.')
        else:
            rows = [{'Rank': c.rank, 'Candidate': candidate_label(c),
                     'Evidence-adjusted match': score(c.ranking_score),
                     'Score coverage': percentage(c.score_coverage),
                     'Required-skill coverage': percentage(c.required_skill_coverage),
                     'Evidence': ', '.join(f'{k}: {v}' for k, v in c.evidence_strength_summary.items()) or 'Not assessed'}
                    for c in filtered.filtered_candidates]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            candidates = {c.candidate_id: c for c in filtered.filtered_candidates}
            if st.session_state.get('selected_candidate') not in candidates:
                st.session_state['selected_candidate'] = next(iter(candidates))
            selected = st.selectbox('Inspect candidate', list(candidates), key='selected_candidate',
                                    format_func=lambda key: f'#{candidates[key].rank} — {candidate_label(candidates[key])}')
            record = records[selected]
            show_candidate(candidates[selected], record['profile'], record['filename'], record['classification'])

        show_comparison(ranking)
