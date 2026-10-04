"""Foundation checks; no tests write into the project's output/model directories."""
import contextlib
import hashlib
import io
import json
import os
import pickle
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from src.main import build_models, load_dataset, normalize_resume_key, run_experiment, split_dataset
from src.preprocessing import clean_resume
from src.predict import hybrid_predict

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/raw/resume_dataset.csv'


class FoundationTests(unittest.TestCase):
    def test_technical_terms_and_determinism(self):
        text = 'C++ C# .NET Python JavaScript TypeScript Node.js React.js ASP.NET'
        expected = 'cplusplus csharp dotnet python javascript typescript nodejs reactjs aspnet'
        self.assertEqual(clean_resume(text), expected)
        self.assertEqual(clean_resume(text), clean_resume(text))
        tokens = build_models()['Naive Bayes'].named_steps['tfidf'].build_analyzer()(text)
        for term in expected.split():
            self.assertIn(term, tokens)

    def test_contact_noise(self):
        self.assertEqual(clean_resume('Python https://example.com/cpp me@example.com +91 98765 43210'), 'python')
        self.assertEqual(clean_resume('JavaScript www.example.com (987) 654-3210'), 'javascript')
        self.assertEqual(clean_resume('Python 3.9 experience 2020'), 'python 3 9 experience 2020')

    def test_duplicate_keys_preserve_punctuation(self):
        self.assertEqual(normalize_resume_key('  PYTHON\nDev '), 'python dev')
        self.assertEqual(len({normalize_resume_key(x) for x in ['C', 'C++', 'C#', '.NET']}), 4)

    def test_validation_and_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.csv'
            cases = [({'Resume':['x']}, 'Missing required columns'),
                     ({'Category':['A']}, 'Missing required columns'),
                     ({'Resume':[' '], 'Category':['A']}, 'missing/blank resumes=1'),
                     ({'Resume':[None], 'Category':['A']}, 'missing/blank resumes=1'),
                     ({'Resume':['x'], 'Category':[None]}, 'missing/blank labels=1'),
                     ({'Resume':['x'], 'Category':[' ']}, 'missing/blank labels=1'),
                     ({'Resume':[' X ', 'x'], 'Category':['A','B']}, 'Conflicting duplicate labels')]
            for data, message in cases:
                pd.DataFrame(data).to_csv(path,index=False)
                with self.assertRaisesRegex(ValueError, message), contextlib.redirect_stdout(io.StringIO()):
                    load_dataset(path)
            pd.DataFrame({'Resume':[' X ', 'x', 'C++', 'C#'], 'Category':['A']*4}).to_csv(path,index=False)
            before = path.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                result = load_dataset(path)
            self.assertEqual(result.Resume.tolist(), [' X ', 'C++', 'C#'])
            self.assertEqual(path.read_bytes(), before)

    def test_population_split_and_source_integrity(self):
        before = hashlib.sha256(DATA.read_bytes()).hexdigest()
        with contextlib.redirect_stdout(io.StringIO()):
            data = load_dataset(DATA)
        train, test = split_dataset(data)
        self.assertEqual((len(data),len(train),len(test)), (166,132,34))
        self.assertEqual(data.Category.nunique(),25)
        self.assertFalse(set(train.Resume.map(normalize_resume_key)) & set(test.Resume.map(normalize_resume_key)))
        self.assertEqual(set(train.Category), set(test.Category))
        self.assertEqual(hashlib.sha256(DATA.read_bytes()).hexdigest(),before)

    def test_unseen_text_never_enters_vocabulary(self):
        for model in build_models().values():
            model.fit(['python django']*2 + ['java spring']*2, ['A','A','B','B'])
            vocabulary = dict(model.named_steps['tfidf'].vocabulary_)
            model.predict(['testonlysentinel testonlysentinel'])
            self.assertNotIn('testonlysentinel', vocabulary)
            self.assertEqual(vocabulary, model.named_steps['tfidf'].vocabulary_)

    def test_fresh_process_raw_text_pipeline(self):
        model = build_models()['LinearSVC'].fit(['C++ systems']*2 + ['Python django']*2,['A','A','B','B'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.pkl'
            path.write_bytes(pickle.dumps(model))
            code = 'import pickle,sys,json; m=pickle.load(open(sys.argv[1],"rb")); print(json.dumps(m.predict(["C++ systems"]).tolist()))'
            env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1')
            result = subprocess.run([sys.executable,'-B','-c',code,str(path)],cwd=directory,env=env,capture_output=True,text=True,check=True)
            self.assertEqual(json.loads(result.stdout),model.predict(['C++ systems']).tolist())

    def test_no_fake_confidence_or_override(self):
        model=build_models()['LinearSVC'].fit(['java spring']*2+['python django']*2,['A','A','B','B'])
        prediction, confidence, top3, method = hybrid_predict('python django',model)
        self.assertEqual(prediction, model.predict(['python django'])[0])
        self.assertEqual((confidence,top3,method),(None,[],'ML'))

    def test_keyword_override_has_no_probability(self):
        class LowConfidenceModel:
            classes_ = np.array(['A','B'])
            def predict(self,texts): return np.array(['A'])
            def predict_proba(self,texts): return np.array([[.51,.49]])
        with patch('src.predict.keyword_predict',return_value=('Python Developer',2)):
            self.assertEqual(hybrid_predict('python django',LowConfidenceModel()),('Python Developer',None,[],'keyword'))

    def test_experiment_reproducibility(self):
        with contextlib.redirect_stdout(io.StringIO()):
            _, first, report1 = run_experiment(DATA)
            _, second, report2 = run_experiment(DATA)
        self.assertEqual(first,second)
        self.assertEqual(report1,report2)
        self.assertEqual(first['duplicate_key_overlap'],0)
        self.assertEqual(first['cv']['n_splits'],2)
        held_out=set(first['split']['test_row_indices'])
        for fold in first['cv']['fold_membership']:
            self.assertFalse(held_out & set(fold['train_row_indices']))
            self.assertFalse(held_out & set(fold['validation_row_indices']))
            self.assertFalse(set(fold['train_row_indices']) & set(fold['validation_row_indices']))


if __name__ == '__main__':
    unittest.main()
