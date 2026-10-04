"""Evidence rules tested with synthetic text; no inference of truth from wording."""
import copy
import io
import unittest
from unittest.mock import patch
from pathlib import Path
from src.evidence import assess_candidate
from src.job_intelligence import JobProfile, parse_job_description
from src.resume_intelligence import parse_resume
from src.resume_intelligence.models import CandidateProfile, Experience, Project
from src.matching import match_candidate_to_job

ROOT = Path(__file__).resolve().parents[1]


def assessment(text, required=None):
    return assess_candidate(parse_resume(text), JobProfile(required_skills=required or ['Python']))


class EvidenceTests(unittest.TestCase):
    def test_skills_only_weak(self):
        result=assessment('Skills\nPython')
        self.assertEqual(result.required_requirements[0].status,'weak_evidence')
        self.assertEqual(result.evidence_adjusted_match_score,30)

    def test_project_moderate(self):
        result=assessment('Projects\nBuilt backend services using Python.')
        self.assertEqual(result.required_requirements[0].evidence_strength,'moderate')

    def test_experience_strong(self):
        result=assessment('Experience\nDeveloped backend services using Python for 2 years.')
        self.assertEqual(result.required_requirements[0].evidence_strength,'strong')

    def test_certification_supporting(self):
        result=assessment('Certifications\nAWS Certified Solutions Architect',['AWS'])
        self.assertEqual(result.required_requirements[0].evidence_strength,'moderate')
        self.assertEqual(result.required_requirements[0].evidence[0].source_type,'certification')
        self.assertEqual(result.overall_match_score,0)
        self.assertEqual(result.evidence_adjusted_match_score,70)

    def test_isolated_keyword_in_experience_not_strong(self):
        self.assertEqual(assessment('Experience\nPython').required_requirements[0].evidence_strength,'weak')

    def test_indirect_team_context(self):
        result=assessment('Experience\nWorked with a team where Python was one of many technologies.')
        self.assertEqual(result.required_requirements[0].evidence_strength,'weak')

    def test_keyword_stuffing_comparison(self):
        job=['Python','FastAPI','PostgreSQL','Docker']
        text='Skills\nPython, FastAPI, PostgreSQL, Docker'
        weak=assessment(text,job)
        strong=assessment(text+'\nExperience\nDeveloped production REST APIs using Python, FastAPI and PostgreSQL.\nProjects\nContainerized the service with Docker.',job)
        self.assertEqual(weak.overall_match_score,strong.overall_match_score)
        self.assertEqual(weak.evidence_adjusted_match_score,30)
        self.assertEqual(strong.evidence_adjusted_match_score,92.5)

    def test_repetition_does_not_raise_score(self):
        once=assessment('Skills\nPython')
        repeated=assessment('Skills\n'+'Python\n'*15)
        self.assertEqual(once.evidence_adjusted_match_score,repeated.evidence_adjusted_match_score)
        self.assertEqual(len(repeated.required_requirements[0].evidence),1)
        self.assertTrue(any('repetition' in concern for concern in repeated.concerns))
        self.assertLess(repeated.evidence_adjusted_match_score,
                        assessment('Experience\nDeveloped backend services using Python for 2 years.').evidence_adjusted_match_score)

    def test_missing(self):
        item=assessment('Skills\nJava').required_requirements[0]
        self.assertEqual(item.status,'missing')
        self.assertFalse(item.direct_match)
        self.assertIsNone(item.evidence_strength)

    def test_structured_skill_no_context(self):
        result=assess_candidate(CandidateProfile(skills=['Python']),JobProfile(required_skills=['Python']))
        self.assertEqual(result.required_requirements[0].status,'present_no_context')
        self.assertEqual(result.required_requirements[0].evidence[0].source_text,'Python')

    def test_flask_transferable(self):
        item=assessment('Experience\nDeveloped REST APIs using Flask.',['FastAPI']).required_requirements[0]
        self.assertEqual(item.status,'transferable')
        self.assertTrue(item.transferable)
        self.assertFalse(item.direct_match)
        self.assertTrue(all(e.relevance=='transferable' for e in item.evidence))

    def test_azure_transferable(self):
        result=assessment('Experience\nDeployed cloud infrastructure using Azure.',['AWS'])
        self.assertTrue(result.required_requirements[0].transferable)
        self.assertFalse(result.required_requirements[0].direct_match)
        self.assertEqual(result.evidence_adjusted_match_score,0)

    def test_original_snippets_and_deduplication(self):
        text='Developed backend services using Python.'
        candidate=CandidateProfile(skills=['Python'],experience=[Experience(description=text),Experience(description=text)],raw_text='Experience\n'+text)
        item=assess_candidate(candidate,JobProfile(required_skills=['Python'])).required_requirements[0]
        self.assertEqual(len(item.evidence),1)
        self.assertEqual(item.evidence[0].source_text,text)

    def test_strongest_source_preferred(self):
        item=assessment('Skills\nPython\nProjects\nBuilt tools using Python.\nExperience\nMaintained services using Python.').required_requirements[0]
        self.assertEqual(item.evidence_strength,'strong')
        self.assertEqual(item.evidence[0].strength,'strong')
        self.assertEqual({e.source_type for e in item.evidence},{'skills','project','experience'})

    def test_relevant_experience(self):
        result=assess_candidate(parse_resume('Experience\nDeveloped backend APIs using Python.'),
                                JobProfile(experience_requirements=['2+ years backend development using Python']))
        self.assertEqual(result.experience_relevance[0].status,'relevant')
        self.assertEqual(result.match_result.experience_match,'unknown')

    def test_partial_experience(self):
        result=assess_candidate(parse_resume('Experience\nCreated Python data analysis notebooks.'),
                                JobProfile(experience_requirements=['2+ years backend development using Python']))
        self.assertEqual(result.experience_relevance[0].status,'partial')

    def test_unknown_experience(self):
        result=assess_candidate(CandidateProfile(),JobProfile(experience_requirements=['2+ years backend development']))
        self.assertEqual(result.experience_relevance[0].status,'unknown')
        self.assertEqual(result.experience_relevance[0].source_texts,[])

    def test_summary_not_employment(self):
        self.assertEqual(assessment('Summary\nDeveloped services using Python.').required_requirements[0].evidence_strength,'weak')

    def test_no_fake_evidence_from_evidence_dictionary(self):
        candidate=CandidateProfile(evidence={'skills.Python':'Developed Python APIs in production'})
        item=assess_candidate(candidate,JobProfile(required_skills=['Python'])).required_requirements[0]
        self.assertEqual(item.evidence,[])
        self.assertFalse(item.direct_match)

    def test_negation_and_hypothetical(self):
        for text in ['No experience with Python.', 'Planning to build services using Python.', 'I would develop APIs using Python.']:
            item=assessment('Experience\n'+text).required_requirements[0]
            self.assertEqual(item.status,'unknown')
            self.assertFalse(item.direct_match)
            self.assertEqual(item.evidence[0].source_text,text)

    def test_clauses_do_not_borrow_actions(self):
        item=assessment('Experience\nDeveloped Java services; Python.').required_requirements[0]
        self.assertEqual(item.evidence_strength,'weak')

    def test_literal_job_concepts(self):
        result=assessment('Projects\nBuilt a FastAPI backend with JWT authentication.', ['FastAPI','JWT','authentication','backend','Docker'])
        self.assertEqual([item.status for item in result.required_requirements],['evidenced']*4+['missing'])
        self.assertTrue(all(item.evidence_strength=='moderate' for item in result.required_requirements[:4]))

    def test_strengths_weaknesses_concerns_explanation(self):
        result=assessment('Skills\nDocker\nExperience\nDeveloped Python services.',['Python','Docker','AWS'])
        self.assertTrue(result.strengths)
        self.assertTrue(result.weaknesses)
        self.assertTrue(result.concerns)
        self.assertIn('not probabilities',result.explanation)
        self.assertIn('supporting',result.required_requirements[1].explanation)

    def test_deterministic_non_mutating_baseline(self):
        candidate=parse_resume('Skills\nPython\nProjects\nBuilt Python tools.')
        job=JobProfile(required_skills=['Python'])
        originals=copy.deepcopy((candidate,job))
        result=assess_candidate(candidate,job)
        self.assertEqual((candidate,job),originals)
        self.assertEqual(result.to_dict(),assess_candidate(candidate,job).to_dict())
        self.assertEqual(result.match_result,match_candidate_to_job(candidate,job))

    def test_empty_job_score_unknown(self):
        result=assess_candidate(CandidateProfile(),JobProfile())
        self.assertIsNone(result.evidence_adjusted_match_score)
        self.assertEqual(result.required_requirements,[])

    def test_aliases_and_technical_boundaries(self):
        result=assessment('Skills\nPostgres, C++, C#',['PostgreSQL','C','C++','C#'])
        self.assertEqual([item.direct_match for item in result.required_requirements],[True,False,True,True])

    def test_only_job_requirements_assessed(self):
        result=assessment('Skills\nPython, Java, Docker')
        self.assertEqual([item.requirement for item in result.required_requirements],['Python'])

    def test_preferred_evidence(self):
        result=assess_candidate(parse_resume('Projects\nContainerized services with Docker.'),
                                JobProfile(preferred_skills=['Docker','AWS']))
        self.assertEqual(result.preferred_requirements[0].evidence_strength,'moderate')
        self.assertEqual(result.evidence_adjusted_match_score,35)

    def test_streamlit_evidence_view(self):
        from streamlit.testing.v1 import AppTest
        upload=io.BytesIO(b'Skills\nPython\nExperience\nDeveloped backend services using Python.')
        upload.name='resume.txt'
        with patch('streamlit.file_uploader',return_value=[upload]):
            app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=30)
            app.text_area[0].set_value('Required Skills: Python').run(timeout=30)
            app.button[0].click().run(timeout=30)
            self.assertFalse(app.exception)
            self.assertTrue(any('Evidence assessment' in item.label for item in app.expander))
            self.assertTrue(any('Developed backend' in item.value for item in app.text))


if __name__=='__main__':
    unittest.main()
