"""Tests use synthetic profiles; no training or external data is needed."""
import unittest
from src.resume_intelligence.models import CandidateProfile, Education, Experience
from src.job_intelligence import JobProfile, parse_job_description
from src.resume_intelligence import parse_resume
from src.matching import match_candidate_to_job


def example():
    candidate=CandidateProfile(
        skills=['Python','FastAPI','PostgreSQL','Docker'],
        education=[Education(degree='B.Tech',field='Computer Science')],
        experience=[Experience(role='Software Engineer',description='3 years Software Engineer')])
    job=JobProfile(required_skills=['Python','FastAPI','PostgreSQL'],
                   preferred_skills=['Docker','AWS'],
                   education_requirements=["Bachelor's degree in Computer Science"],
                   experience_requirements=['2+ years backend development'])
    return candidate,job


class MatchingTests(unittest.TestCase):
    def test_all_required(self):
        result=match_candidate_to_job(*example())
        self.assertEqual(result.matched_required_skills,['Python','FastAPI','PostgreSQL'])
        self.assertEqual(result.missing_required_skills,[])
        self.assertEqual(result.required_skill_match_ratio,1.0)

    def test_missing_and_ratio(self):
        candidate,job=example(); job.required_skills.append('AWS')
        result=match_candidate_to_job(candidate,job)
        self.assertEqual(result.missing_required_skills,['AWS'])
        self.assertEqual(result.required_skill_match_ratio,.75)

    def test_no_required(self):
        result=match_candidate_to_job(CandidateProfile(),JobProfile())
        self.assertIsNone(result.required_skill_match_ratio)
        self.assertIsNone(result.overall_match_score)
        self.assertEqual(result.score_coverage,0)

    def test_preferred(self):
        result=match_candidate_to_job(*example())
        self.assertEqual(result.matched_preferred_skills,['Docker'])
        self.assertEqual(result.missing_preferred_skills,['AWS'])
        self.assertEqual(result.preferred_skill_match_ratio,.5)

    def test_no_preferred(self):
        result=match_candidate_to_job(CandidateProfile(skills=['Python']),JobProfile(required_skills=['Python']))
        self.assertIsNone(result.preferred_skill_match_ratio)
        self.assertEqual(result.overall_match_score,100)
        self.assertEqual(result.score_coverage,.7)

    def test_aliases_deduplication_and_no_semantics(self):
        result=match_candidate_to_job(CandidateProfile(skills=['Postgres','scikit learn','JavaScript','C++','Java EE']),
                                     JobProfile(required_skills=['PostgreSQL','postgres','scikit-learn','Java','C','C#']))
        self.assertEqual(result.matched_required_skills,['PostgreSQL','scikit-learn'])
        self.assertEqual(result.missing_required_skills,['Java','C','C#'])
        self.assertEqual(result.required_skill_match_ratio,.4)

    def test_experience_satisfied(self):
        self.assertEqual(match_candidate_to_job(*example()).experience_match,'satisfied')

    def test_experience_unknown(self):
        candidate,job=example(); candidate.experience=[]
        self.assertEqual(match_candidate_to_job(candidate,job).experience_match,'unknown')
        candidate.experience=[Experience(description='Maintained 5 years of records')]
        self.assertEqual(match_candidate_to_job(candidate,job).experience_match,'unknown')

    def test_experience_below_threshold(self):
        candidate,job=example(); candidate.experience=[Experience(description='1 year Software Engineer')]
        result=match_candidate_to_job(candidate,job)
        self.assertEqual(result.experience_match,'not_satisfied')
        self.assertEqual(result.overall_match_score,82.5)

    def test_dates_overlap_and_open_ends(self):
        candidate,job=example()
        candidate.experience=[Experience(start_date='2020-01-01',end_date='2023-01-01')]*2
        job.experience_requirements=['4 years experience']
        self.assertEqual(match_candidate_to_job(candidate,job).experience_match,'unknown')
        job.experience_requirements=['2+ years experience']
        self.assertEqual(match_candidate_to_job(candidate,job).experience_match,'satisfied')
        candidate.experience=[Experience(start_date='2020',end_date='Present')]
        self.assertEqual(match_candidate_to_job(candidate,job).experience_match,'unknown')

    def test_year_only_dates_are_conservative(self):
        candidate,job=example()
        candidate.experience=[Experience(start_date='2020',end_date='2022')]
        self.assertEqual(match_candidate_to_job(candidate,job).experience_match,'unknown')

    def test_skill_specific_tenure_unknown(self):
        candidate,job=example(); job.experience_requirements=['2 years of Python']
        self.assertEqual(match_candidate_to_job(candidate,job).experience_match,'unknown')

    def test_education_satisfied(self):
        self.assertEqual(match_candidate_to_job(*example()).education_match,'satisfied')

    def test_education_unknown(self):
        candidate,job=example(); candidate.education=[]
        self.assertEqual(match_candidate_to_job(candidate,job).education_match,'unknown')
        candidate.education=[Education(degree='B.Tech',field='Physics')]
        self.assertEqual(match_candidate_to_job(candidate,job).education_match,'unknown')

    def test_optional_or_ambiguous_requirements_unknown(self):
        candidate,job=example()
        job.experience_requirements=['2-5 years experience']
        job.education_requirements=["Bachelor's degree or equivalent experience"]
        result=match_candidate_to_job(candidate,job)
        self.assertEqual(result.experience_match,'unknown')
        self.assertEqual(result.education_match,'unknown')

    def test_score_and_determinism(self):
        first=match_candidate_to_job(*example())
        self.assertEqual(first.overall_match_score,92.5)
        self.assertEqual(first.score_coverage,1.0)
        self.assertEqual(first.to_dict(),match_candidate_to_job(*example()).to_dict())

    def test_unknown_not_zero(self):
        candidate,job=example(); candidate.experience=[]; candidate.education=[]
        result=match_candidate_to_job(candidate,job)
        self.assertEqual(result.overall_match_score,91.18)
        self.assertEqual(result.score_coverage,.85)

    def test_explanation(self):
        result=match_candidate_to_job(*example())
        self.assertIn('3/3',result.explanation)
        self.assertIn('1/2',result.explanation)
        self.assertIn('Duration-only',result.explanation)
        self.assertIn('Rule-based match score',result.explanation)

    def test_evidence_only_from_profile(self):
        candidate,job=example()
        self.assertEqual(match_candidate_to_job(candidate,job).matched_skill_evidence,{})
        candidate.evidence={'skills.Postgres':'Skills: Postgres', 'skills.AWS':'Other text'}
        self.assertEqual(match_candidate_to_job(candidate,job).matched_skill_evidence,
                         {'PostgreSQL':'Skills: Postgres'})

    def test_parser_integration(self):
        candidate=parse_resume('Skills: Python, FastAPI, Postgres, Docker\nEducation\nB.Tech in Computer Science\nExperience\n3 years Software Engineer')
        job=parse_job_description("Required Skills: Python, FastAPI, Postgres\nPreferred Skills: Docker, AWS\nEducation\nBachelor's degree in Computer Science\nExperience\n2+ years backend development")
        self.assertEqual(match_candidate_to_job(candidate,job).overall_match_score,92.5)


if __name__ == '__main__':
    unittest.main()
