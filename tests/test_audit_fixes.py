"""Focused finalization regressions; no changes to scoring weights or ranking."""
import pytest

from src.evidence import assess_candidate
from src.job_intelligence import JobProfile, parse_job_description
from src.matching import match_candidate_to_job
from src.resume_intelligence import parse_resume
from src.resume_intelligence.models import CandidateProfile, Project


@pytest.mark.parametrize('sentence', [
    'I have never used AWS.', 'I have not used AWS.', "I haven't used AWS.",
    'I have no experience with AWS.', 'I am without AWS experience.',
    'I am not experienced in AWS.', 'I am learning AWS.',
    'I am currently learning AWS.', 'I am interested in AWS.', 'I am familiar with AWS.',
])
@pytest.mark.parametrize('section', ['Projects', 'Experience'])
def test_unsupported_context_has_no_credit(section, sentence):
    result = assess_candidate(parse_resume(section + '\n' + sentence), JobProfile(required_skills=['AWS']))
    requirement = result.required_requirements[0]
    assert requirement.status == 'unknown'
    assert not requirement.direct_match
    assert result.evidence_adjusted_match_score == 0
    assert requirement.evidence
    assert not any(e.supports_requirement for e in requirement.evidence)
    assert any(e.source_text == sentence for e in requirement.evidence)


@pytest.mark.parametrize('qualifier', ['listed Python as an interest', 'am learning Python',
                                     'have not used Python', 'am not experienced in Python',
                                     'listed Python as unrelated', 'listed Python as knowledge only'])
def test_mixed_clause_does_not_borrow_java_action(qualifier):
    result = assess_candidate(parse_resume('Experience\nBuilt Java services and ' + qualifier + '.'),
                              JobProfile(required_skills=['Python', 'Java']))
    python, java = result.required_requirements
    assert not python.direct_match
    assert python.status == 'unknown'
    assert java.evidence_strength == 'strong'


@pytest.mark.parametrize('sentence,skills', [
    ('Built AWS services using Python.', ['AWS', 'Python']),
    ('Built backend services using Python and FastAPI.', ['Python', 'FastAPI']),
])
def test_legitimate_joint_skill_use_is_strong(sentence, skills):
    result = assess_candidate(parse_resume('Experience\n' + sentence), JobProfile(required_skills=skills))
    assert all(r.direct_match and r.evidence_strength == 'strong' for r in result.required_requirements)
    assert result.evidence_adjusted_match_score == 100


def test_positive_project_and_repetition_remain_supported():
    job = JobProfile(required_skills=['AWS'])
    once = assess_candidate(parse_resume('Projects\nBuilt AWS services.'), job)
    repeated = assess_candidate(parse_resume('Projects\n' + 'Built AWS services.\n' * 10), job)
    assert once.evidence_adjusted_match_score == repeated.evidence_adjusted_match_score == 70


def test_structured_project_without_description_keeps_mention():
    result = assess_candidate(CandidateProfile(projects=[Project(technologies=['AWS'])]), JobProfile(required_skills=['AWS']))
    assert result.required_requirements[0].evidence_strength == 'weak'


@pytest.mark.parametrize('heading', ['Preferred', 'Preferred:', 'Preferred Skills:', 'Nice to Have:',
                                     'Nice-to-Have Skills:', 'Good to Have:'])
def test_preferred_heading_ends_required_context(heading):
    job = parse_job_description('Required Skills:\nPython\n' + heading + '\nAWS')
    assert job.required_skills == ['Python']
    assert job.preferred_skills == ['AWS']


@pytest.mark.parametrize('heading', ['Required Skills:', 'Required:', 'Must Have:', 'Must-Have Skills:'])
def test_required_heading_variants(heading):
    job = parse_job_description(heading + '\nPython\nPreferred:\nAWS')
    assert job.required_skills == ['Python']
    assert job.preferred_skills == ['AWS']


def test_ambiguous_heading_stops_required_context():
    job = parse_job_description('Required:\nPython\nOther Skills\nAWS')
    assert job.required_skills == ['Python']
    assert job.preferred_skills == []
    assert 'AWS' in job.keywords


def test_original_job_fixture_requirements_preserved():
    text = '''We are looking for a Backend Engineer to build and maintain reliable backend services and REST APIs.
Required skills:
Python
FastAPI
PostgreSQL
Preferred skills:
Docker
AWS
Redis
Responsibilities:
Build and maintain REST APIs using Python and FastAPI.
Design and maintain PostgreSQL databases.
Develop reliable backend services.
Write clean, maintainable and testable code.
Deploy and maintain services using cloud/container technologies.
Education:
Bachelor's degree in Computer Science, Information Technology, or a related field.
Experience:
2+ years of backend software development experience.'''
    job = parse_job_description(text)
    assert job.required_skills == ['Python', 'FastAPI', 'PostgreSQL']
    assert job.preferred_skills == ['Docker', 'AWS', 'Redis']
    assert len(job.responsibilities) == 5
    assert job.education_requirements == ["Bachelor's degree in Computer Science, Information Technology, or a related field."]
    assert job.experience_requirements == ['2+ years of backend software development experience.']


@pytest.mark.parametrize('heading', ['Experience', 'Experience Details:', 'Professional Experience',
                                    'Work Experience', 'Employment', 'Employment History', 'Career Experience'])
def test_professional_heading_after_skills(heading):
    text = 'Name: Rahul Sharma\nSkills:\nPython, FastAPI\n' + heading + '\nBuilt production REST APIs using Python and FastAPI.'
    profile = parse_resume(text)
    assert profile.experience
    assert profile.raw_text == text
    result = assess_candidate(profile, JobProfile(required_skills=['Python', 'FastAPI']))
    assert all(r.evidence_strength == 'strong' for r in result.required_requirements)
    assert all(any(e.source_type == 'experience' for e in r.evidence) for r in result.required_requirements)


@pytest.mark.parametrize('degree', ['B.Tech Computer Science', 'B.Tech in Computer Science', 'B.Tech Computer Science.', 'B.Tech in Computer Science.'])
def test_bare_cs_requirement_matches_cs(degree):
    job = parse_job_description('Education:\nB.Tech Computer Science')
    result = match_candidate_to_job(parse_resume('Education\n' + degree), job)
    assert result.education_match == 'satisfied'


def test_mechanical_degree_does_not_satisfy_cs():
    job = parse_job_description('Education:\nB.Tech Computer Science')
    result = match_candidate_to_job(parse_resume('Education\nB.Tech in Mechanical Engineering'), job)
    # Existing model treats insufficient/nonmatching education as unknown, not a new rejection state.
    assert result.education_match == 'unknown'


@pytest.mark.parametrize('requirement', ['B.Tech from a recognized university', 'B.Tech with minimum 70%',
                                        'B.Tech meeting our accreditation criteria'])
def test_unparseable_restriction_not_silently_removed(requirement):
    job = parse_job_description('Education:\n' + requirement)
    result = match_candidate_to_job(parse_resume('Education\nB.Tech in Computer Science'), job)
    assert result.education_match == 'unknown'


def test_no_education_requirement_unchanged():
    assert match_candidate_to_job(parse_resume(''), JobProfile()).education_match == 'not_required'


def test_degree_only_requirement_still_works():
    result = match_candidate_to_job(parse_resume('Education\nB.Tech in Computer Science'),
                                    JobProfile(education_requirements=["Bachelor's degree"]))
    assert result.education_match == 'satisfied'
