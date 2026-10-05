# Resume Intelligence & Candidate Evaluation Platform

A local recruiter prototype for extracting candidate profiles, evaluating resume evidence
against job requirements, ranking/filtering candidates, and comparing 2–4 candidates.
Classification into 25 job categories is one component; it does not determine candidate fit.
The evaluation workflow is deterministic and uses no external AI service.

```text
Resume files → Document Extraction → CandidateProfile
Job description → Job Intelligence → JobProfile
CandidateProfile + JobProfile → Candidate ↔ Job Matching → Evidence Assessment
                             → Ranking + Filtering → Recruiter Dashboard → Candidate Comparison
```

PDF, DOCX and TXT are supported. Optional OCR supports scanned PDFs and images using
local dependencies. Complex layouts, DOCX tables and mixed text/scanned PDF pages can
lose content; the tool does not handle every layout automatically or verify resume claims.

---

## 🌟 Key Features

- **Multi-Format Extraction:** Automatically extracts text from `.pdf`, `.docx`, and `.txt`.
- **Optional local OCR:** Scanned documents and images use Tesseract and pdf2image when the Python packages and system tools are installed.
- **Classification pipeline:** Uses TF-IDF with a classifier selected by training-only validation. Models with probabilities can use the existing keyword fallback below its threshold; LinearSVC returns an ML prediction with confidence shown as N/A. The keyword fallback is not part of the ML benchmark.
- **Folder batching:** Classify readable input documents into category CSV files; these operational files may contain candidate information.

---

## 🛠 Project Structure

```text
app.py                         # Streamlit recruiter workflow
src/
  main.py, preprocessing.py    # Reproducible ML experiment
  document_extraction.py       # Shared local document readers
  predict.py, classify_resumes.py
  resume_intelligence/         # CandidateProfile extraction
  job_intelligence/            # JobProfile extraction
  matching/                    # Candidate/job rule-based match
  evidence/                    # Evidence-aware assessment
  ranking/                     # Deterministic ranking and filtering
  comparison/                  # Compare existing evaluated results
  recruiter_ui.py              # Presentation helpers
tests/                         # Full pytest regression suite
data/raw/resume_dataset.csv    # Original benchmark data
models/                        # Local generated model, ignored by Git
output/                        # Intentional reports plus operational outputs
requirements.txt               # Core dependencies
requirements-ocr.txt           # Optional OCR Python packages
```

---

## 🚀 How To Install & Run

The canonical tested environment is **Python 3.9.6** in **`venv`**. For this existing
checkout, reuse it; do not recreate the environment or retrain a model merely to launch:

```bash
cd /Users/Ankush/Desktop/resume-classifier
source venv/bin/activate
python --version                   # expected: Python 3.9.6
python -m streamlit run app.py
```

For a fresh checkout, use the setup below. Verify the interpreter version first;
do not substitute an arbitrary global Python. Training is needed only when the trusted
local `models/model.pkl` artifact is absent or an intentional new experiment is desired.

### 1. Install System Requirements (For OCR Image Scanning)
Optional OCR requires both the Python packages in `requirements-ocr.txt` and the Poppler/Tesseract system tools. Text PDFs, DOCX and TXT do not need OCR.

**Mac:**
```bash
brew install tesseract poppler
```

**Windows:**
1. Install [Tesseract OCR for Windows](https://github.com/UB-Mannheim/tesseract/wiki).
2. Download [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases/), extract it, and add the `Library\bin` or `bin` folder to your system's PATH.

*(Ubuntu: `sudo apt-get install tesseract-ocr poppler-utils`)*

### 2. Set Up Python Environment (fresh checkout only)

Choose an interpreter that reports **Python 3.9.6**. On the original Mac, it is
`/Library/Developer/CommandLineTools/usr/bin/python3`; confirm before using it:

```bash
/Library/Developer/CommandLineTools/usr/bin/python3 --version
/Library/Developer/CommandLineTools/usr/bin/python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install pytest==8.4.2
```

On another machine, replace that executable with the absolute path to your verified
Python 3.9.6 installation. On Windows activate with `venv\Scripts\activate`.
The core ML versions are pinned in `requirements.txt`; the checked UI environment uses
Streamlit 1.50.0, pdfplumber 0.11.8 and python-docx 1.2.0. Unpinned packages are not a
complete environment lock.

If you need OCR, after activating the environment run:

```bash
python -m pip install -r requirements-ocr.txt
tesseract --version
pdftoppm -v
```

No OCR package installation is needed for text-only testing. Installation alone does
not establish extraction accuracy on your scanned documents.

### 3. Train the Model 🧠
Run this only if a model is missing or you intentionally want to repeat the experiment. It compares three pipelines using training-only two-fold cross-validation, selects by macro-F1, then evaluates once on held-out resumes and saves the complete pipeline. Legacy 100% results were affected by duplicate and TF-IDF leakage and are not valid unseen-resume performance.
```bash
python src/main.py   # Use python3 on Mac/Linux
```

### 4. Batch Classify Real Resumes 📂
**Option A (Folder Mode):** Simply drag and drop all your PDFs, Word files, and Images directly into the `data/input/resumes/` folder!
**Option B (CSV Mode):** If the folder is empty, the code dynamically falls back to the `resumes_to_classify.csv` file automatically.

Then, run:
```bash
python src/classify_resumes.py   # Use python3 on Mac/Linux
```
*Results will automatically generate beautifully separated CSVs in the `output/` folder.*

### 5. Check a Single File Interactively 🔍
Need to check just one candidate quickly? Run:
```bash
python src/predict.py   # Use python3 on Mac/Linux
```
It will open an interactive prompt. You can **paste a file path** (`/Users/You/resume.pdf` or `.png`) directly, or just copy and paste raw resume text to instantly see the predicted category and available model confidence—not an accuracy measurement!

### 6. Launch the Web App (Streamlit UI) 🌐
> A trusted `models/model.pkl` must exist. Reuse the existing artifact; follow Step 3 only if it is missing.

```bash
streamlit run app.py
```
This opens the recruiter dashboard: enter a job, upload resumes, evaluate, filter and inspect candidates.

---

## 📌 Troubleshooting
- **`File does not exist` / `Invalid value` error in Streamlit:** The model hasn't been trained yet. Run `python src/main.py` first to generate `models/model.pkl`, then re-run `streamlit run app.py`.
- **`EmptyDataError` when running `main.py`:** Make sure your `data/raw/resume_dataset.csv` file actually has data in it and isn't 0 bytes! (You can type `git restore data/raw/resume_dataset.csv` if you accidentally cleared it).
- **File skipped because of "zlib / corrupted" error:** Sometimes downloaded PDFs are physically corrupted (zero text layer and compressed incorrectly). Open the file in an application like Mac's *Preview* app or a built-in PDF reader on Windows, select `Print` or `Export as PDF`, and save a fresh copy. Then the script will read it immediately.


## Stage 0 experiment

Use Python 3.9.6 with the pinned ML dependencies. Run from the repository root:

```bash
pytest -q
python src/main.py
```

The raw CSV stays unchanged. Duplicate-key normalization is separate from model
preprocessing. After deduplication, raw text is split 80/20 with seed 42 and category
stratification. Each two-fold validation fit learns its own TF-IDF vocabulary.
Selection uses mean macro-F1, then accuracy, then weighted-F1, then alphabetical
model name. Only the selected pipeline is refitted on training data and evaluated
on test; test data is never used for fitting or selection.

`output/experiment_metadata.json` records dataset hash, original row indices for
splits/folds, settings, versions and full metrics. `output/classification_report.txt`
and `output/confusion_matrix.csv` describe the current experiment. Previous models
and reports are archived under `output/legacy/` before replacement.

`models/model.pkl` accepts raw strings and requires this repository's `src` package
on the Python import path. Run consumers from the repository root as above. LinearSVC
has no probabilities: confidence is displayed as N/A and this alone never triggers
keyword fallback. Keyword overrides also display N/A and are outside the ML benchmark.

There are few unique resumes per category; a small held-out result is preliminary.
The recorded 88.24% accuracy is **30/34 held-out test examples**, not production accuracy
and not validation of candidate ranking. Exact/normalized duplicate separation does not establish independence of near-duplicate
templates or prove real-world candidate performance. Tests repeat the fixed experiment
only to verify reproducibility, not to tune against test results.

## Stage 1: Resume document intelligence

```text
Resume file → shared document extraction → CandidateProfile → existing ML classification
```

The public API runs locally and deterministically, without an LLM or external API:

```python
from src.resume_intelligence import parse_resume, parse_resume_file

profile = parse_resume_file("resume.pdf")  # PDF, DOCX, TXT, PNG, JPG, JPEG
# Or: profile = parse_resume("Name: Asha Rao\nSkills: Python, PostgreSQL")
print(profile.to_dict())
```

`CandidateProfile` contains name, email, phone, labeled location, LinkedIn/GitHub/
portfolio links, education, experience, projects, skills, certifications,
achievements, raw extracted text, and lightweight source snippets. Missing fields
are `None` or empty lists. The original text is retained; model preprocessing is
never applied to it. Non-string parser input returns an empty profile. File reading
errors raise `DocumentExtractionError`; the UI and command-line tools report them.

Rules operate on recognized section headings and preserve ambiguous entry text.
Names require an explicit `Name:` label or a short name-shaped header near contact
information; headings, role titles and contact values are rejected. Uncertain names
stay unavailable. Locations require an explicit label. Dates are retained as written;
no dates, experience duration, institutions or employers are inferred.

Skills and aliases live in `src/resume_intelligence/skills.json`. Extend that file
rather than scattering keywords through parser code. Long aliases take precedence:
C, C++, C#, .NET and ASP.NET remain distinct. C/R require uppercase tokens in a skills
context; Go requires that context or the alias Golang. Skills are **text mentions**,
not verified proficiency, endorsements or evidence of suitability for a job.

`src/document_extraction.py` consolidates the existing readers. Line breaks are
preserved for section detection; a small compatibility formatter retains the old
PDF/image/DOCX input shape for classification. TXT decoding replaces invalid bytes.
OCR remains optional and needs the existing pytesseract/pdf2image/Pillow Python
packages plus Tesseract/Poppler system tools. OCR errors are now explicit instead
of silently swallowed. Text PDFs and DOCX/TXT need no OCR installation.

After classification, Streamlit exposes a **Candidate Information** expander for
each readable upload, including the full structured profile and evidence. Existing
classification results remain visible. Uploads use temporary files rather than
writing into the project's output directory.

Run the full automated suite with `pytest -q` (after activating `venv`). Stage 1 parser unit tests
use synthetic text. Local integration tests additionally use the three PDFs under
`pdf/` when available, without modifying them; those local files are not committed.
OCR routes are tested with mocks, not proof of OCR accuracy on scanned documents.
The existing Stage 0 repeatability test fits temporary in-memory models but does not
replace the saved artifact or evaluation reports.

This is a first deterministic parser, not perfect NLP: unusual layouts, DOCX tables,
columns, unlabeled contact data and entries without clear boundaries can remain
partially parsed. Bullet-only projects keep their descriptions with an unavailable
name. There is no job matching, ranking, database, REST API or recruiter workflow.

## Stage 2: Job intelligence foundation

`Job Description → JobProfile` is available locally through:

```python
from src.job_intelligence import parse_job_description
profile = parse_job_description("Required Skills: Python, Postgres\nPreferred Skills: Docker")
print(profile.to_dict())
```

The dataclass contains title, required/preferred skills, responsibilities, education
and experience requirements, canonical technical keywords, and unchanged raw text.
It reuses Stage 1's skills taxonomy and aliases. Explicit wording overrides section
context; ambiguous qualifications, mixed clauses, alternatives and conflicting
skill statuses are not promoted to mandatory requirements. Unclassified technical
mentions remain in `keywords`, which does not imply a requirement. Education and
experience strings retain their original wording, including preference qualifiers.
Missing fields stay null/empty. This conservative first version expects recognizable
headings or explicit requirement wording and can leave unusual prose unclassified.
There is no UI change; candidate matching, scoring and ranking belong to future stages.

## Stage 3: Single-candidate rule-based matching

```python
from src.matching import match_candidate_to_job
result = match_candidate_to_job(candidate_profile, job_profile)
print(result.to_dict())
```

Matching uses canonical exact skills and only aliases already present in Stage 1's
shared taxonomy. Unlisted skill labels require exact case-insensitive equality;
related skills are not substitutes. Missing skills mean **not found in the profile**,
not proven absent. Matched evidence is copied only from existing candidate evidence.

The result includes matched/missing required and preferred skills, their ratios,
experience/education states, an explanation, evidence, score and assessed-weight
coverage. States are `satisfied`, `not_satisfied`, `unknown`, or `not_required`.
Zero requested skills yield a null ratio, not a perfect or failed match.

The **rule-based match score** is 0–100: required skills 70%, preferred skills 15%,
experience 10%, education 5%. Ratios provide skill component values; satisfied is 1
and not_satisfied is 0. Unknown and unspecified components are excluded and remaining
weights are renormalized. With no assessable components the score is null. Coverage
reports the sum of included original weights; 100 points with low coverage does not
mean every requirement was verified. Scores are not AI predictions or hiring decisions.

Experience checks only one obvious years threshold. An explicit duration at the
start of an experience description (e.g. `3 years Software Engineer`) or closed date
intervals can support it. Overlapping dates are merged; separate duration claims are
never summed. Partial dates use conservative bounds; `Present` is unknown rather than
consulting today's date. Skill-specific tenure, alternatives and optional/ambiguous
requirements remain unknown. Satisfaction establishes duration only, not backend or
other role/domain relevance. Insufficient date evidence stays unknown; a single explicit
duration below the threshold is marked not_satisfied.

Education supports a small set of degree-family equivalents (e.g. B.Tech and
Bachelor's) and exact stated field names. Higher degrees are not automatically
substituted. Missing/nonmatching evidence stays unknown; completion and accreditation
are not verified. Original profiles and ML classification behavior are unchanged.

Example: 3/3 required skills, 1/2 preferred skills, satisfied experience and education
produce `70 + 7.5 + 10 + 5 = 92.5`. If experience and education are unknown, that same
skill overlap scores `(70 + 7.5) / 0.85 = 91.18`, with 85% coverage. Stage 3 itself only evaluates one candidate; subsequent stages consume its result.

## Stage 4: Evidence-based candidate intelligence

```text
CandidateProfile + JobProfile → unchanged Stage 3 MatchResult
                            → evidence analysis → CandidateAssessment
```

```python
from src.evidence import assess_candidate
assessment = assess_candidate(candidate_profile, job_profile)
print(assessment.to_dict())
```

**The system does not treat keyword presence as proof of professional experience.**
This is deterministic/rule-based, not an LLM judgement. Even strong evidence describes
resume wording, not independently verified employment, skill level or truthfulness.

`overall_match_score` and `match_result` preserve Stage 3's result. Required/preferred
requirement assessments add status, strongest evidence label, direct/transferable
flags, source snippets, field paths and explanations. Only requested skills/concepts
are assessed; unrelated resume keywords do not improve the score. Existing canonical
skill aliases are reused. Explicit unlisted concepts (e.g. JWT or authentication) can
match literal text, but are never inferred from FastAPI or other framework mentions.

Rules in `src/evidence/rules.py` distinguish:

- **Weak:** lists, summaries, isolated mentions, or indirect team exposure. An
  Experience heading alone does not make a mention strong.
- **Moderate:** a project/academic/achievement snippet contains both a concrete action
  and task, or a certification snippet explicitly mentions a credential.
- **Strong:** an experience snippet contains a concrete work action and task alongside
  the concept. Production experience is never inferred from a project heading.
- **Transferable:** a small explicit backend/database/cloud grouping identifies related
  technologies. Flask is not FastAPI, Azure is not AWS, and SQL is not PostgreSQL.
  Related evidence never counts as a direct match or earns direct scoring credit.
- **Missing:** no supporting mention in supplied text; this does not prove a skill is
  absent. Negated/hypothetical-only mentions are **unknown**, with original text retained.

Statuses are `evidenced`, `weak_evidence`, `present_no_context`, `transferable`,
`missing`, and `unknown`. Structured skill entries without contextual text remain
weak; generated evidence summaries are not trusted as new source text. Snippets come
from raw sections or actual structured fields. Duplicate concept/snippet pairs keep
the strongest source. Repeated keywords never add strength or scoring credit; five
weak-context lines can trigger a neutral verification concern, not an accusation.

The separate **evidence-adjusted match score** retains Stage 3's 70/15/10/5 weights
and available-component normalization. Each requested skill gets its strongest direct
support: strong=1.0, moderate=0.7, weak=0.3; no supporting direct evidence=0.0.
That zero means no evidence credit, not proven inability. Required/preferred component
values are the means across their unique requested skills. Frequency and transferability
add nothing. Experience/education contributions and unknown handling remain exactly
Stage 3's; evidence relevance is reported separately rather than silently overriding
those checks. The score can recover evidence missed by the profile's skill list, e.g.
a certification mention, so it is not necessarily lower than Stage 3's score. Both
scores remain visible, and discrepancies are flagged. Neither score is a probability
or hiring recommendation; unknown/no-requirement components and coverage remain explicit.

Experience relevance checks a small set of backend/frontend/data-analysis/cloud
markers plus explicitly requested skills within the same concrete work snippet.
Python notebooks can partially support a backend-Python requirement without establishing
backend experience. Relevance does not verify duration. No relevant text means unknown.
Education verification remains Stage 3's limited degree/field check. Optional, indirect,
negated or unusual prose may be conservatively missed: these are simple lexical rules,
not semantic understanding or a way to detect deceptive claims.

For four required skills, a list-only resume scores 30 on evidence while retaining
its Stage 3 skill-match score of 100. Three strong work-backed skills plus one moderate
project-backed skill score 92.5. This tests the rules, not differences in real candidates.

Streamlit has an optional job-description text field and a per-resume evidence expander
with both scores, explanations, concerns and verbatim snippets. Existing classification
and profile views remain available. Stage 5 adds batch ranking and filters below these views; no new API or external service is involved.


## Stage 5: Evidence-aware ranking and filtering

```text
CandidateAssessment[] → Ranking + Filtering → Ranked Candidates → Recruiter Review
```

```python
from src.ranking import CandidateFilters, rank_candidates, filter_candidates

# Assessments must concern the same job; profiles are optional identity inputs only.
ranked = rank_candidates(assessments, candidate_profiles=profiles)
shortlist = filter_candidates(ranked, CandidateFilters(
    minimum_score=70, minimum_required_skill_coverage=0.75,
    required_skills=["Python"], minimum_evidence_strength="moderate"))
for candidate in shortlist.filtered_candidates:
    print(candidate.rank, candidate.candidate_name, candidate.ranking_explanation)
for excluded in shortlist.excluded_candidates:
    print(excluded.candidate.candidate_name, excluded.reasons)
```

Ordering is lexicographic: descending **unchanged Stage 4 evidence-adjusted score**,
then direct required-skill coverage, Stage 4 assessed weight coverage, mean strongest
direct evidence credit per unique requested skill (the existing 0.3/0.7/1.0 labels),
then ascending stable identifier. Secondary signals only break ties, never add points.
Repeated evidence records, keywords and unrelated skills add no strength credit.
Unavailable scores sort below known zero scores and remain unavailable. Weight coverage
includes unspecified/unknown components excluded from evaluation; low coverage is not
proof of inability. Required coverage counts direct matches, including weak mentions,
so it must be read alongside evidence strength. Missing, unknown and transferable states
remain separate. Transferable evidence never becomes a direct match.

Filters support minimum score (0–100), minimum required coverage (0–1), selected
canonical skills (existing aliases), experience and education statuses (`satisfied`,
`not_satisfied`, `unknown`, `not_required`), and minimum evidence strength
(`weak`, `moderate`, `strong`). Every selected skill must match directly; for a skill
not assessed by Stage 4, the existing Stage 3 canonical candidate skill list can establish
presence only. It cannot establish evidence strength. A strength filter applies to
**every selected skill**, otherwise every required job skill, falling back to preferred
skills when there are no required skills. No assessed skills means strength is unknown
and cannot pass an explicit threshold. Jobs with no required skills have `None` coverage
and pass the coverage filter as not applicable; incomplete required assessments fail
an explicit coverage threshold. Unknown scores fail even an explicit minimum of zero.

`RankingResult` retains all candidates, the full ranked list, included candidates,
excluded candidates with reasons, and a filter summary. Filtering starts from the full
ranked batch, does not mutate the input, and preserves original rank numbers. Ranked
views retain the original assessment, strengths, concerns, requirement states and evidence.

IDs hash profile content, including normalized email when available, without displaying
contact details. They identify submissions, not verified people; changing profile content
can change the ID. Without profiles the assessment hash is evaluation-specific. Identical
submissions are retained with stable occurrence suffixes and an explicit duplicate count;
they are interchangeable, never silently merged. These hashes are not an anonymization
or identity-verification system. The caller must supply assessments for one job: the
existing assessment model does not contain a job identity that Stage 5 can validate.

Streamlit retains evaluated uploads during filter reruns, displays a compact ranked table,
requirement states, explanations and exclusion reasons, and preserves classification,
profile and evidence views. Changing the job or uploads requires clicking Classify Resumes
again (Stage 6 calls this action Evaluate Candidates). Ranking is decision support, not autonomous hiring or a prediction of job performance;
all evidence remains rule-based information in resumes, not verified claims.

Run all regression tests (pytest is a development-only tool):

```bash
python -m pip install pytest==8.4.2
pytest -q
```

## Stage 6: Recruiter dashboard and end-to-end workflow

Start the local dashboard with `streamlit run app.py`, using the existing saved
classification model and dependencies. No external AI service is required.

```text
Job Description → Multiple Resume Upload → Candidate Evaluation
                → Evidence-Aware Ranking → Filtering → Candidate Detail Review
```

Enter a description and optionally a job title. The existing job parser supplies a
compact requirement summary; the title field labels the job without changing its
requirements. Upload PDF, DOCX, TXT, or supported images; OCR uses the existing local
Stage 1 dependencies. Click **Evaluate Candidates** once both description and resumes
are present. Classification remains available in a results expander, including its
existing CSV download. The CLI classification workflows are unchanged.

The ranked table shows names (or `Candidate N`), evidence-adjusted match scores,
assessed-weight coverage, direct required-skill coverage and the existing evidence
strength summary. Contact details appear only in the selected candidate's detail view.
Filters call Stage 5 directly and expose exclusion reasons. They produce a displayed
subset, not a saved hiring decision or a persistent shortlist.

Select a candidate to inspect the structured profile, classification, Stage 3 matching,
Stage 4 evidence-adjusted score, required/preferred requirement states, verbatim evidence
snippets, experience, education, projects, strengths, concerns and Stage 5 rank explanation.
Transferable evidence is explicitly separate from direct coverage. Missing and unknown
remain distinct. Evidence is information found in the resume; claims are not independently
verified. Scores are deterministic decision support, not hiring probabilities or predictions
of real-world performance.

Session state separates inputs (`input_*`, `job_profile`, `uploaded_resume_metadata`),
evaluation (`evaluation_results`, `ranking_result`, `failed_uploads`, `evaluation_signature`),
filters (`filter_*`) and selection (`selected_candidate`). Filter and selection changes
reuse the stored profiles, classifications, assessments and ranking. They do not rerun
extraction, classification, matching, evidence assessment or ranking. File content hashes
are checked on reruns, so input checking still reads uploaded bytes.

Changing the title, description, filenames, file contents or duplicate-file count clears
evaluation, filter and selection state immediately. Reordering the same upload set does
not invalidate results. Reverting changed inputs does not resurrect an earlier evaluation;
click Evaluate Candidates again. Repeated explicit evaluation requests rerun the batch.
Unsupported, empty and unreadable files are reported individually; processing continues
for other files. Unexpected failures identify the affected processing phase and exception
type without displaying internal exception payloads. Missing/unloadable model and batch
ranking errors remain visible rather than publishing incomplete results.

This remains a local, sequential, in-memory prototype: session loss/restart loses results,
there is no database, authentication, durable shortlist, external API, or background job
processing. Large batches can take time and memory. Stage 0–5 domain rules are unchanged;
`src/recruiter_ui.py` only formats existing data and gathers filter inputs. The dashboard
requires a job before evaluation; the earlier profile-upload UI regression now supplies
that job while keeping its classification/profile assertions.

Run `pytest -q` for the full regression suite. Stage 6 AppTest cases use synthetic
uploads and check the workflow, failure handling, invalidation and service call counts.

## Stage 7: Candidate comparison

After evaluation, select **2–4 candidates** in **Candidate Comparison**. The selector
uses the full evaluated batch, including candidates hidden by filters. Comparison
selection persists during filtering and normal candidate inspection. **Clear comparison**
removes it; changing job/upload inputs or explicitly reevaluating clears it automatically.
Comparison views are rebuilt from stored results, so no stale comparison snapshot is kept.

```text
CandidateAssessment → RankedCandidate → Recruiter Workflow → Candidate Comparison
```

The overview shows original rank, Stage 3 match, Stage 4 evidence-adjusted score,
coverage, and existing experience/education statuses. Required/preferred skill tables
preserve direct, transferable, missing and unknown states and show a representative
resume excerpt where available. Evidence summaries distinguish professional from project
sources. Strengths, concerns, original ranking explanations and deterministic pairwise
observations make differences inspectable. Full evidence remains in candidate inspection.

```python
from src.comparison import compare_candidates
comparison = compare_candidates(ranking_result, selected_candidate_ids)  # 2–4 distinct IDs
```

The API selects from one existing ranked batch and orders columns by original rank.
It retains original objects and values, does not rescore or rerank, and rejects unknown
or repeated identifiers. Separate submissions with distinct Stage 5 IDs remain selectable,
even if their names or resumes match. Missing assessment records are unassessed, not
inferred missing skills. Available source labels are taken from supporting direct evidence;
transferable and negated snippets cannot establish direct professional/project support.
Repetition adds no comparison advantage. A lower-ranked candidate may have better coverage
on one dimension; pairwise observations describe that without overriding the full ranking.

Comparison triggers no extraction, OCR, classification, job parsing, matching, assessment,
or ranking. Stage 6 still checks uploaded content hashes on reruns. No extra dependencies
or external services are required. These are comparisons of deterministic resume-text
assessments, not verified claims, hiring probabilities, or proof of professional competence.
Stage 6 manual recruiter testing remains separate from automated regression/UI tests.


## Finalization corrections and local-data hygiene

Evidence retains negative/hypothetical context instead of reusing extracted project
technology tokens as independent positive claims. Familiarity, interest and knowledge-only
wording do not establish direct use. Limited clause rules keep these qualifications from
borrowing unrelated work actions; they are not a general language-understanding system.
Known job-heading variants end required/preferred context explicitly. Experience Details
and Career Experience are recognized resume headings. Simple bare education fields such
as B.Tech Computer Science are compared literally; unparsed restrictions remain unknown.
A nonmatching degree field retains the existing unknown education state, not an invented
rejection or a claim that the candidate has no other qualifications.

Git ignores real upload folders, `pdf/`, generated classification CSVs, logs, caches,
local secrets and model archives. It intentionally does **not** ignore all of `output/`:
experiment metadata, current/historical evaluation reports and archive notes remain
reviewable. Ignore rules do not untrack existing files. Already-tracked operational CSVs,
logs and `.DS_Store` should be considered separately for a controlled cleanup; nothing
is automatically deleted. Keep the tracked input sample only as a deliberate synthetic
fixture, never replace it with private resumes and commit it. Resume data is not anonymized
by using hashed candidate identifiers.
