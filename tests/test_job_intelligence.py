import unittest
from src.job_intelligence import JobProfile, parse_job_description

SAMPLE = """Job Title: Backend Engineer
Responsibilities
- Build REST APIs using Python.
- Maintain reliable services.
Requirements
- Python, FastAPI and Postgres
- 2+ years experience building backend services
- Bachelor's degree in Computer Science
Preferred Skills
- Docker and scikit learn
"""


class JobTests(unittest.TestCase):
    def test_title(self):
        self.assertEqual(parse_job_description(SAMPLE).title, 'Backend Engineer')
        self.assertEqual(parse_job_description('Senior Python Developer\nRequirements\nPython').title,'Senior Python Developer')
        self.assertIsNone(parse_job_description('Welcome to our team').title)

    def test_required_skills(self):
        self.assertEqual(parse_job_description(SAMPLE).required_skills,['Python','FastAPI','PostgreSQL'])
        self.assertEqual(parse_job_description('Python is mandatory').required_skills,['Python'])

    def test_preferred_skills(self):
        self.assertEqual(parse_job_description(SAMPLE).preferred_skills,['Docker','scikit-learn'])
        for wording in ['preferred','nice to have','a plus','bonus']:
            self.assertEqual(parse_job_description('Docker is '+wording).preferred_skills,['Docker'])

    def test_aliases_and_boundaries(self):
        profile=parse_job_description('Required Skills: Postgres, PostgreSQL, scikit learn, C++, C#, .NET')
        self.assertEqual(profile.required_skills,['C++','C#','.NET','PostgreSQL','scikit-learn'])
        self.assertNotIn('C',profile.required_skills)

    def test_responsibilities(self):
        profile=parse_job_description(SAMPLE)
        self.assertEqual(profile.responsibilities,['Build REST APIs using Python.','Maintain reliable services.'])
        self.assertNotIn('REST API',profile.required_skills)
        self.assertEqual(parse_job_description("What You’ll Do:\n• Build services").responsibilities,['Build services'])

    def test_education(self):
        self.assertEqual(parse_job_description(SAMPLE).education_requirements,["Bachelor's degree in Computer Science"])
        self.assertEqual(parse_job_description('Education: MBA preferred').education_requirements,['MBA preferred'])

    def test_experience(self):
        self.assertEqual(parse_job_description(SAMPLE).experience_requirements,['2+ years experience building backend services'])
        self.assertEqual(parse_job_description('Experience\n3 years of Python').experience_requirements,['3 years of Python'])

    def test_missing_sections_and_ambiguity(self):
        profile=parse_job_description('We use Python and Postgres.')
        self.assertEqual(profile.keywords,['Python','PostgreSQL'])
        self.assertEqual(profile.required_skills,[])
        self.assertEqual(profile.preferred_skills,[])
        self.assertEqual(parse_job_description('Qualifications: Python').required_skills,[])

    def test_negation_alternatives_and_conflicts(self):
        for text in ['Requirements\nPython is not required', 'Required skills: Python or Java',
                     'Python required and Docker preferred', 'Required Skills: Python\nPreferred Skills: Python']:
            profile=parse_job_description(text)
            self.assertEqual(profile.required_skills,[])
            self.assertEqual(profile.preferred_skills,[])
        self.assertEqual(parse_job_description('Education\nNo degree required').education_requirements,[])

    def test_inline_override_and_section_boundary(self):
        profile=parse_job_description('Requirements\nPython required; Docker preferred\nBenefits\nFree Java courses')
        self.assertEqual(profile.required_skills,['Python'])
        self.assertEqual(profile.preferred_skills,['Docker'])
        self.assertNotIn('Java',profile.required_skills)

    def test_empty_malformed_no_invention(self):
        for value in [None,42,{},[],b'bad','', '\x00', 'Friendly workplace']:
            profile=parse_job_description(value)
            self.assertIsInstance(profile,JobProfile)
            self.assertIsNone(profile.title)
            for field in ['required_skills','preferred_skills','education_requirements','experience_requirements','responsibilities']:
                self.assertEqual(getattr(profile,field),[])

    def test_raw_text_and_determinism(self):
        self.assertEqual(parse_job_description(SAMPLE).raw_text,SAMPLE)
        self.assertEqual(parse_job_description(SAMPLE).to_dict(),parse_job_description(SAMPLE).to_dict())


if __name__ == '__main__':
    unittest.main()
