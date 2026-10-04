"""Streamlit presentation only; domain services own all evaluation decisions."""
from dataclasses import asdict
from typing import List, Optional

import streamlit as st

from src.evidence.models import RequirementAssessment
from src.job_intelligence.models import JobProfile
from src.ranking import CandidateFilters, RankedCandidate, RankingResult
from src.resume_intelligence.models import CandidateProfile


def score(value: Optional[float]) -> str:
    return 'Not assessable' if value is None else f'{value:.1f} / 100'


def percentage(value: Optional[float]) -> str:
    return 'Not applicable / unavailable' if value is None else f'{value:.0%}'


def candidate_label(candidate: RankedCandidate) -> str:
    return candidate.candidate_name or f'Candidate {candidate.rank}'


def show_job(job: JobProfile) -> None:
    with st.expander('Parsed job summary', expanded=True):
        st.write('Job:', job.title or 'Not detected')
        for label, values in [('Required skills', job.required_skills),
                              ('Preferred skills', job.preferred_skills),
                              ('Responsibilities', job.responsibilities),
                              ('Experience requirements', job.experience_requirements),
                              ('Education requirements', job.education_requirements)]:
            st.write(label, values or 'Not detected')
        if not (job.required_skills or job.preferred_skills or
                job.experience_requirements or job.education_requirements):
            st.info('No assessable requirements detected. Use clear requirement headings or wording in the description.')


def show_filters(ranking: RankingResult) -> CandidateFilters:
    skills = sorted({skill for candidate in ranking.ranked_candidates
                     for skill in (candidate.assessment.match_result.candidate_skills +
                                   [r.requirement for r in candidate.assessment.required_requirements +
                                    candidate.assessment.preferred_requirements])})
    with st.expander('Filter candidates'):
        minimum_score = st.number_input('Minimum evidence-adjusted score', min_value=0.0,
                                       max_value=100.0, value=None, key='filter_score')
        coverage = st.number_input('Minimum required-skill coverage (0–1)', min_value=0.0,
                                   max_value=1.0, value=None, key='filter_coverage')
        selected = st.multiselect('Require direct skills', skills, key='filter_skills')
        statuses = ['Any', 'satisfied', 'not_satisfied', 'unknown', 'not_required']
        experience = st.selectbox('Experience status', statuses, key='filter_experience')
        education = st.selectbox('Education status', statuses, key='filter_education')
        strength = st.selectbox('Minimum evidence strength', ['Any', 'weak', 'moderate', 'strong'],
                                key='filter_strength')
        st.caption('Strength applies to every selected skill; otherwise every required job skill, '
                   'or preferred skills when none are required. Unknown and transferable are not direct evidence.')
    return CandidateFilters(minimum_score, coverage, selected,
                            None if experience == 'Any' else experience,
                            None if education == 'Any' else education,
                            None if strength == 'Any' else strength)


def show_requirements(label: str, requirements: List[RequirementAssessment]) -> None:
    st.subheader(label)
    if not requirements:
        st.write('No requirements specified.')
    for requirement in requirements:
        # These are existing labels, not a fresh inference from resume text.
        st.write(f'{requirement.requirement} — {requirement.status}')
        st.write('Direct match:', requirement.direct_match,
                 'Evidence strength:', requirement.evidence_strength or 'Not assessable')
        if requirement.transferable:
            st.info('Transferable evidence detected. It does not count as direct skill coverage.')
        st.write(requirement.explanation)


def show_candidate(candidate: RankedCandidate, profile: CandidateProfile, filename: str,
                   classification: dict) -> None:
    assessment = candidate.assessment
    match = assessment.match_result
    st.header('Selected candidate')
    st.subheader(candidate_label(candidate))
    st.caption(f'Rank #{candidate.rank} · Source: {filename}')
    with st.expander('Candidate Overview', expanded=True):
        st.write({label: value or 'Not detected' for label, value in {
            'Name': profile.candidate_name, 'Location': profile.location,
            'Email': profile.email, 'Phone': profile.phone}.items()})
        st.write('Links', {key: value or 'Not detected' for key, value in asdict(profile.links).items()})
        st.write('Skills', profile.skills or 'Not detected')
        st.write('Classification', classification)

    st.subheader('Match Summary')
    columns = st.columns(3)
    columns[0].metric('Evidence-adjusted match', score(candidate.ranking_score))
    columns[1].metric('Score coverage', percentage(candidate.score_coverage))
    columns[2].metric('Direct required-skill coverage', percentage(candidate.required_skill_coverage))
    st.write('Stage 3 rule-based match:', score(assessment.overall_match_score))
    st.write('Stage 3 preferred-skill coverage:', percentage(match.preferred_skill_match_ratio))
    st.caption('The evidence-adjusted score uses Stage 4 evidence. Coverage is assessed requirement weight, '
               'not a hiring probability or proof that claims are true.')
    st.write('Evidence strength summary', candidate.evidence_strength_summary)
    show_requirements('Required Skills', assessment.required_requirements)
    show_requirements('Preferred Skills', assessment.preferred_requirements)

    with st.expander('Evidence assessment — evidence found in resume', expanded=True):
        evidence_found = False
        for requirement in assessment.required_requirements + assessment.preferred_requirements:
            for evidence in requirement.evidence:
                evidence_found = True
                st.write(requirement.requirement)
                st.caption(f'{evidence.strength} · {evidence.source_type} · {evidence.section} · {evidence.relevance}')
                if not evidence.supports_requirement:
                    st.caption('This snippet does not establish the requirement: ' + evidence.reason)
                if evidence.relevance == 'transferable':
                    st.caption('Transferable evidence — not counted as direct coverage.')
                st.text(evidence.source_text)
        if not evidence_found:
            st.write('No evidence snippets detected for this job.')
        st.caption('Resume claims have not been independently verified.')

    for label, records, status, detail in [
        ('Experience', profile.experience, candidate.experience_status, match.experience_detail),
        ('Education', profile.education, candidate.education_status, match.education_detail),
    ]:
        with st.expander(label):
            st.write('Requirement status:', status)
            st.write(detail)
            if records:
                for record in records:
                    st.write({k: v or 'Not detected' for k, v in asdict(record).items()})
            else:
                st.write('Not detected')
            if label == 'Experience':
                for relevance in assessment.experience_relevance:
                    st.write(asdict(relevance))
    with st.expander('Projects, certifications and achievements'):
        st.write('Projects', [asdict(p) for p in profile.projects] or 'Not detected')
        st.write('Certifications', profile.certifications or 'Not detected')
        st.write('Achievements', profile.achievements or 'Not detected')
    st.subheader('Strengths')
    for strength in candidate.strengths:
        st.write('• ' + strength)
    if not candidate.strengths:
        st.write('No strengths reported by the assessment.')
    st.subheader('Concerns')
    for concern in candidate.concerns + assessment.weaknesses:
        st.write('• ' + concern)
    if not (candidate.concerns or assessment.weaknesses):
        st.write('No concerns reported by these rules; claims still require review.')
    st.subheader(f'Why ranked #{candidate.rank}')
    st.write(candidate.ranking_explanation)
    with st.expander('Candidate Information — structured profile'):
        st.json(profile.to_dict())
