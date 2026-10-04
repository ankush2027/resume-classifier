"""Synthetic parser tests and document/inference compatibility checks."""
import hashlib
import io
import json
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from src.document_extraction import DocumentExtractionError, classification_text, extract_text
from src.resume_intelligence import CandidateProfile, parse_resume, parse_resume_file
from src.resume_intelligence.parser import detect_sections, extract_skills
from src.predict import hybrid_predict

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = '''RESUME
Asha Rao
asha.rao@example.com
Phone: +91 98765 43210
Location: Pune, India
linkedin.com/in/asha-rao
https://github.com/asha-rao
Portfolio: asha.example.org
SUMMARY
Backend engineer working with Python.
TECHNICAL SKILLS
Python, FastAPI, postgres, PostgreSQL, C++, C#, .NET, JavaScript, C, R, Go
EDUCATION
B.Tech in Computer Science
Example Institute of Technology
2018 - 2022
WORK EXPERIENCE
Software Engineer at Example Labs | Jan 2023 - Present
- Built REST APIs with Python and PostgreSQL.
PROJECTS
Resume Reader
- Developed a local parser using Python and FastAPI.
Technologies: Python, PostgreSQL
CERTIFICATIONS
- Example Cloud Certificate
AWARDS
- University coding award
INTERESTS
Hiking
'''


class ParserTests(unittest.TestCase):
    def test_email_phone_location(self):
        profile = parse_resume(SAMPLE)
        self.assertEqual(profile.email, 'asha.rao@example.com')
        self.assertEqual(profile.phone, '+91 98765 43210')
        self.assertEqual(profile.location, 'Pune, India')
        self.assertIsNone(parse_resume('Experience\n2020 - 2024').phone)

    def test_links(self):
        links = parse_resume(SAMPLE).links
        self.assertEqual(links.linkedin, 'https://linkedin.com/in/asha-rao')
        self.assertEqual(links.github, 'https://github.com/asha-rao')
        self.assertEqual(links.portfolio, 'https://asha.example.org')
        profile = parse_resume('Website: https://notgithub.com/person')
        self.assertIsNone(profile.links.github)

    def test_name_heuristic(self):
        self.assertEqual(parse_resume(SAMPLE).candidate_name,'Asha Rao')
        self.assertIsNone(parse_resume('RESUME\nSUMMARY\nPython developer').candidate_name)
        self.assertIsNone(parse_resume('a@example.com\n9876543210').candidate_name)
        self.assertIsNone(parse_resume('Software Engineer\na@example.com').candidate_name)
        self.assertIsNone(parse_resume('Software Development\na@example.com').candidate_name)
        self.assertIsNone(parse_resume('Unknown Person').candidate_name)
        self.assertEqual(parse_resume('Name: Asha Rao').candidate_name, 'Asha Rao')

    def test_skills_and_aliases(self):
        skills = parse_resume(SAMPLE).skills
        for skill in ['Python','FastAPI','PostgreSQL','C++','C#','.NET','JavaScript','C','R','Go']:
            self.assertIn(skill,skills)
        self.assertEqual(skills.count('PostgreSQL'),1)
        self.assertEqual(extract_skills('postgres POSTGRESQL sklearn nodejs react.js'),
                         ['Node.js','PostgreSQL','scikit-learn','React'])

    def test_technical_boundaries(self):
        self.assertEqual(extract_skills('C++ C# .NET',True),['C++','C#','.NET'])
        self.assertEqual(extract_skills('accounting cream research sharper scorn',True),[])
        self.assertEqual(extract_skills('JavaScript TypeScript'),['JavaScript','TypeScript'])
        self.assertEqual(extract_skills('ASP.NET'),['ASP.NET'])
        self.assertEqual(extract_skills('go to work; R is a grade; vitamin C'),[])
        self.assertEqual(extract_skills('C, R, Go',True),['C','Go','R'])

    def test_section_variants_and_inline_heading(self):
        sections = detect_sections('WORK EXPERIENCE:\nOne\nProfessional Experience\nTwo\nSkills: Python, SQL\nPublications\nA paper')
        self.assertIn('One',sections['experience'])
        self.assertIn('Two',sections['experience'])
        self.assertEqual(sections['skills'],'Python, SQL')
        self.assertEqual(sections['publications'],'A paper')

    def test_education(self):
        entry=parse_resume(SAMPLE).education[0]
        self.assertEqual(entry.degree,'B.Tech')
        self.assertEqual(entry.field,'Computer Science')
        self.assertEqual(entry.institution,'Example Institute of Technology')
        self.assertEqual((entry.start_date,entry.end_date),('2018','2022'))
        self.assertEqual(parse_resume('Education\nI hope to be successful.').education,[])
        for degree in ['B.E.','B.Sc','BCA','M.Tech','M.E.','M.Sc','MCA','MBA','PhD']:
            self.assertEqual(len(parse_resume('Education\n'+degree).education),1)

    def test_experience_and_uncertainty(self):
        entry=parse_resume(SAMPLE).experience[0]
        self.assertEqual((entry.role,entry.company),('Software Engineer','Example Labs'))
        self.assertEqual((entry.start_date,entry.end_date),('Jan 2023','Present'))
        self.assertIn('Built REST APIs',entry.description)
        ambiguous=parse_resume('Experience\nWorked on several internal tools.').experience[0]
        self.assertIsNone(ambiguous.company)
        self.assertIsNone(ambiguous.role)
        self.assertIsNone(ambiguous.start_date)
        self.assertEqual(ambiguous.description,'Worked on several internal tools.')

    def test_projects(self):
        project=parse_resume(SAMPLE).projects[0]
        self.assertEqual(project.name,'Resume Reader')
        self.assertEqual(project.technologies,['Python','FastAPI','PostgreSQL'])
        ambiguous=parse_resume('Projects\n- Built a Python script').projects[0]
        self.assertIsNone(ambiguous.name)
        self.assertEqual(ambiguous.technologies,['Python'])

    def test_multiple_entries(self):
        profile=parse_resume('Projects\nProject: Alpha\n- Python\nProject: Beta\n- Java\nEducation\nBCA\n2015 - 2018\nMCA\n2018 - 2020')
        self.assertEqual([p.name for p in profile.projects],['Alpha','Beta'])
        self.assertEqual([e.degree for e in profile.education],['BCA','MCA'])

    def test_certifications_achievements(self):
        profile=parse_resume(SAMPLE)
        self.assertEqual(profile.certifications,['Example Cloud Certificate'])
        self.assertEqual(profile.achievements,['University coding award'])
        self.assertNotIn('Hiking',profile.achievements)

    def test_source_and_evidence_determinism(self):
        profile=parse_resume(SAMPLE)
        self.assertEqual(profile.raw_text,SAMPLE)
        self.assertEqual(profile.to_dict(),parse_resume(SAMPLE).to_dict())
        self.assertIn('Python',profile.evidence['skills.Python'])
        self.assertEqual(json.loads(json.dumps(profile.to_dict()))['raw_text'],SAMPLE)

    def test_empty_missing_malformed(self):
        for value in [None,42,[],{},b'bad','', '\x00\ufffd', 'https://[invalid', 'https://[x]:bad', 'Summary\n...']:
            profile=parse_resume(value)
            self.assertIsInstance(profile,CandidateProfile)
            self.assertIsNone(profile.candidate_name)
            self.assertEqual(profile.experience,[])
            self.assertIsNone(profile.location)
        self.assertEqual(parse_resume('Skills: Python').education,[])
        self.assertIsNone(parse_resume('Skills: Python').email)


class DocumentTests(unittest.TestCase):
    def test_txt_docx_preserve_layout(self):
        from docx import Document
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'resume.txt'; path.write_text(SAMPLE)
            self.assertEqual(parse_resume_file(path).raw_text,SAMPLE)
            doc=Document()
            for line in SAMPLE.splitlines(): doc.add_paragraph(line)
            path=Path(directory)/'resume.docx'; doc.save(path)
            profile=parse_resume_file(path)
            self.assertEqual(profile.candidate_name,'Asha Rao')
            self.assertEqual(profile.projects[0].name,'Resume Reader')
            self.assertIn('\n',profile.raw_text)

    def test_explicit_document_errors(self):
        with self.assertRaises(DocumentExtractionError): extract_text('resume.exe')
        with self.assertRaises(DocumentExtractionError): extract_text('missing.txt')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'broken.pdf'; path.write_text('not a PDF')
            with self.assertRaises(DocumentExtractionError): extract_text(path)

    def test_ocr_routes_and_errors(self):
        import src.document_extraction as readers
        fake_pdf=MagicMock()
        fake_pdf.__enter__.return_value.pages=[MagicMock()]
        fake_pdf.__enter__.return_value.pages[0].extract_text.return_value=''
        image=MagicMock()
        with patch.object(readers,'OCR_SUPPORT',True), patch.object(readers,'PDF_SUPPORT',True), \
             patch.object(readers.pdfplumber,'open',return_value=fake_pdf), \
             patch.object(readers,'convert_from_path',return_value=[image],create=True), \
             patch.object(readers,'pytesseract',create=True) as ocr:
            ocr.image_to_string.return_value=SAMPLE
            self.assertEqual(parse_resume_file('scan.pdf').candidate_name,'Asha Rao')
            image.close.assert_called_once()
            ocr.image_to_string.side_effect=RuntimeError('OCR unavailable')
            with self.assertRaisesRegex(DocumentExtractionError,'OCR unavailable'):
                extract_text('scan.pdf')

    def test_image_classifier_format_compatibility(self):
        text='A\nPython developer'
        self.assertEqual(classification_text(text,'scan.png'),'Python developer')
        self.assertEqual(classification_text(text,'scan.png',clean_images=False),text)

    def test_image_route(self):
        import src.document_extraction as readers
        with patch.object(readers,'OCR_SUPPORT',True), \
             patch.object(readers,'Image',create=True), \
             patch.object(readers,'pytesseract',create=True) as ocr:
            ocr.image_to_string.return_value=SAMPLE
            self.assertEqual(parse_resume_file('scan.png').raw_text,SAMPLE)


class IntegrationTests(unittest.TestCase):
    def test_existing_repository_pdfs_and_classifier(self):
        paths=sorted((ROOT/'pdf').glob('*.pdf'))
        if len(paths)<3:
            self.skipTest('Local example PDFs are not tracked; synthetic document tests remain portable.')
        model_path=ROOT/'models/model.pkl'
        before=hashlib.sha256(model_path.read_bytes()).hexdigest()
        with model_path.open('rb') as stream: model=pickle.load(stream)
        # Compare against pre-Stage-1 extraction rules directly, not against new code alone.
        import pdfplumber
        import re
        for path in paths[:3]:
            profile=parse_resume_file(path)
            self.assertTrue(profile.raw_text.strip())
            self.assertTrue(profile.skills)
            self.assertIsNotNone(profile.email)
            with pdfplumber.open(path) as pdf:
                old_text=''.join((page.extract_text() or '')+'\n' for page in pdf.pages)
            legacy=' '.join(line.strip() for line in old_text.split('\n') if len(line.strip())>=3 and not re.match(r'^\d+$',line.strip()))
            new_text=classification_text(profile.raw_text,path)
            self.assertEqual(legacy,new_text)
            self.assertEqual(hybrid_predict(legacy,model),hybrid_predict(new_text,model))
        self.assertEqual(hashlib.sha256(model_path.read_bytes()).hexdigest(),before)

    def test_streamlit_upload_and_profile(self):
        from streamlit.testing.v1 import AppTest
        upload=io.BytesIO(SAMPLE.encode()); upload.name='resume.txt'
        with patch('streamlit.file_uploader',return_value=[upload]):
            app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=30)
            self.assertFalse(app.exception)
            app.text_area[0].set_value('Required Skills: Python').run(timeout=30)
            app.button[0].click().run(timeout=30)
            self.assertFalse(app.exception)
            self.assertTrue(app.dataframe)
            values=[json.loads(element.value) for element in app.get('json')]
            self.assertTrue(any(isinstance(value,dict) and value.get('candidate_name')=='Asha Rao' for value in values))

    def test_batch_csv_and_documents(self):
        for mode in ['csv', 'documents']:
            with tempfile.TemporaryDirectory() as directory:
                base=Path(directory)
                (base/'models').mkdir()
                shutil.copy2(ROOT/'models/model.pkl',base/'models/model.pkl')
                (base/'data/input').mkdir(parents=True)
                if mode=='csv':
                    shutil.copy2(ROOT/'data/input/resumes_to_classify.csv',base/'data/input/resumes_to_classify.csv')
                else:
                    (base/'data/input/resumes').mkdir()
                    (base/'data/input/resumes/sample.txt').write_text(SAMPLE)
                result=subprocess.run([sys.executable,'-B',str(ROOT/'src/classify_resumes.py')],
                                      cwd=base,capture_output=True,text=True,check=True)
                self.assertIn('classified',result.stdout)
                self.assertTrue(list((base/'output').glob('*_resumes.csv')))

    def test_imports_have_no_side_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            code = ('import sys; sys.path.insert(0, '+repr(str(ROOT))+'); '
                    'import src.main, src.predict, src.classify_resumes; '
                    'from src.resume_intelligence import parse_resume, parse_resume_file')
            result=subprocess.run([sys.executable,'-B','-c',code],cwd=directory,
                                  capture_output=True,text=True,check=True)
            self.assertEqual(result.stdout,'')
            self.assertEqual(list(Path(directory).iterdir()),[])

    def test_cli_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'sample.txt'; path.write_text(SAMPLE)
            result=subprocess.run([sys.executable,'-B','src/predict.py'],cwd=ROOT,
                                  input=str(path)+'\nexit\n',text=True,capture_output=True,check=True)
            self.assertIn('Predicted Category',result.stdout)
            self.assertIn('N/A',result.stdout)


if __name__=='__main__':
    unittest.main()
