"""Conservative section/phrase rules, sharing Stage 1's canonical skill matcher."""
import re
from src.resume_intelligence.parser import DEGREE, extract_skills
from .models import JobProfile

SECTIONS = {
    'requirements': 'required', 'required skills': 'required', 'must have': 'required',
    'mandatory': 'required', 'minimum qualifications': 'required',
    'qualifications': 'qualifications', 'skills': 'unknown',
    'preferred skills': 'preferred', 'preferred qualifications': 'preferred',
    'nice to have': 'preferred', 'bonus': 'preferred',
    'responsibilities': 'responsibilities', "what you'll do": 'responsibilities',
    'what you will do': 'responsibilities', 'key responsibilities': 'responsibilities',
    'experience': 'experience', 'education': 'education',
    'about us': 'other', 'about the company': 'other', 'benefits': 'other',
    'our stack': 'other', 'tech stack': 'other', 'how to apply': 'other',
}
REQUIRED = re.compile(r'\b(?:required|must(?:[- ]have)?|mandatory|essential)\b', re.I)
PREFERRED = re.compile(r'\b(?:preferred|nice[- ]to[- ]have|bonus|a plus|optional|desirable)\b', re.I)
NEGATED = re.compile(r"\b(?:not|no|without|unnecessary|needn[’']t|don[’']t)\b", re.I)
EXPERIENCE = re.compile(r'\b\d+(?:\s*[-–]\s*\d+)?\+?\s+years?\b', re.I)
TITLE_ROLE = re.compile(r'\b(?:engineer|developer|analyst|scientist|manager|designer|architect|intern|consultant)\b', re.I)


def _append_unique(destination, values):
    for value in values:
        if value not in destination:
            destination.append(value)


def _heading(value):
    return re.sub(r'\s+', ' ', value.strip().strip('#* ').rstrip(':').lower().replace('’', "'").replace('-', ' '))


def parse_job_description(text) -> JobProfile:
    """Extract explicit requirements; ambiguous skill mentions remain keywords only.

    Qualifications alone do not establish mandatory status. Mixed/negated clauses
    and skill alternatives are not promoted to mandatory skill lists. Original
    education/experience wording is retained, including preferred qualifiers.
    """
    profile = JobProfile(raw_text=text if isinstance(text, str) else '')
    if not isinstance(text, str) or not text.strip():
        return profile
    section = 'header'
    first_content = True
    for source in text.splitlines():
        line = re.sub(r'^\s*(?:[-*•]\s+|\d+[.)]\s+)', '', source).strip()
        if not line:
            continue
        label = re.match(r'^(?:Job title|Title|Position|Role)\s*:\s*(.+)$', line, re.I)
        if label:
            if profile.title is None:
                profile.title = label.group(1).strip()
            first_content = False
            continue
        heading, colon, content = line.partition(':')
        key = SECTIONS.get(_heading(heading if colon else line))
        if key:
            section = key
            first_content = False
            if not colon or not content.strip():
                continue
            line = content.strip()
        elif line.endswith(':') or (line.isupper() and len(line.split()) <= 5 and not extract_skills(line)):
            # Unknown headings terminate context rather than inheriting 'required'.
            section = 'unknown'
            first_content = False
            continue
        if first_content:
            if TITLE_ROLE.search(line) and len(line.split()) <= 8 and not re.search(r'[.!?;]', line):
                profile.title = line
                first_content = False
                continue
            first_content = False
        if section == 'responsibilities':
            _append_unique(profile.responsibilities, [line])
        # Semicolons separate explicit independent clauses; do not split technical dots.
        for clause in re.split(r'\s*;\s*', line):
            required, preferred = bool(REQUIRED.search(clause)), bool(PREFERRED.search(clause))
            status = ('required' if required else 'preferred') if required != preferred else None
            if not required and not preferred and section in {'required', 'preferred'}:
                status = section
            explicit_skills = section in {'required', 'preferred', 'qualifications'} or required or preferred
            skills = extract_skills(clause, explicit_skills=explicit_skills)
            _append_unique(profile.keywords, skills)
            negated = bool(NEGATED.search(clause))
            alternatives = len(skills) > 1 and bool(re.search(r'\bor\b', clause, re.I))
            if status and not negated and not alternatives:
                _append_unique(getattr(profile, status + '_skills'), skills)
            requirement_context = (section in {'required', 'preferred', 'qualifications', 'education', 'experience'}
                                   or required or preferred)
            if not negated and requirement_context:
                if DEGREE.search(clause):
                    _append_unique(profile.education_requirements, [clause])
                if EXPERIENCE.search(clause):
                    _append_unique(profile.experience_requirements, [clause])
    # Conflicting statements are uncertain, not silently resolved in either direction.
    conflicting = set(profile.required_skills) & set(profile.preferred_skills)
    profile.required_skills = [skill for skill in profile.required_skills if skill not in conflicting]
    profile.preferred_skills = [skill for skill in profile.preferred_skills if skill not in conflicting]
    return profile
