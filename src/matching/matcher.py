"""Canonical skill overlap plus limited, explicitly scoped requirement checks."""
from datetime import date, datetime
import re
from src.job_intelligence.models import JobProfile
from src.job_intelligence.parser import NEGATED, PREFERRED
from src.resume_intelligence.models import CandidateProfile
from src.resume_intelligence.parser import DEGREE, TAXONOMY, extract_skills
from .models import MatchResult

WEIGHTS = {'required': 0.70, 'preferred': 0.15, 'experience': 0.10, 'education': 0.05}
YEARS = re.compile(r'(?<![\w.-])(\d+(?:\.\d+)?)\+?\s+years?\b', re.I)


def _skills(values):
    """Use the existing matcher; retain unlisted skills as exact casefolded labels."""
    result = []
    aliases = {name: [name.casefold()] + [alias.casefold() for alias in variants]
               for category in TAXONOMY.values() for name, variants in category.items()}
    for value in values:
        if not isinstance(value, str) or not value.strip():
            continue
        normalized = ' '.join(value.casefold().split())
        canonical = [name for name in extract_skills(value, explicit_skills=True)
                     if normalized in aliases[name]]
        for name in canonical or [normalized]:
            if name not in result:
                result.append(name)
    return result


def _uncertain_wording(text):
    return bool(NEGATED.search(text) or PREFERRED.search(text) or
                re.search(r'\b(?:or|between|up to|less than|under)\b|\d\s*[-–]\s*\d', text, re.I))


def _date_bound(value, start):
    """Year/month-only dates use latest possible start and earliest possible end."""
    if not value:
        return None
    value = value.strip()
    if re.fullmatch(r'\d{4}', value):
        year = int(value)
        return date(year, 12, 31) if start else date(year, 1, 1)
    for fmt in ['%Y-%m-%d', '%b %Y', '%B %Y', '%m/%Y', '%Y-%m']:
        try:
            parsed = datetime.strptime(value, fmt).date()
        except ValueError:
            continue
        if fmt != '%Y-%m-%d' and start:
            import calendar
            parsed = parsed.replace(day=calendar.monthrange(parsed.year, parsed.month)[1])
        return parsed
    # Present is unknown without an explicit, fixed endpoint; never consult the clock.
    return None


def _experience(candidate, requirements):
    if not requirements:
        return 'not_required', 'No experience requirement supplied.'
    if len(requirements) != 1 or _uncertain_wording(requirements[0]):
        return 'unknown', 'Experience wording is optional, ambiguous, or contains multiple requirements.'
    matches = YEARS.findall(requirements[0])
    if len(matches) != 1:
        return 'unknown', 'No single supported years-of-experience threshold.'
    # Specific skill tenure cannot be established by generic employment dates.
    if extract_skills(requirements[0], explicit_skills=True):
        return 'unknown', 'Skill-specific experience duration cannot be verified from general employment.'
    threshold = float(matches[0])
    explicit, intervals = [], []
    for entry in candidate.experience:
        durations = YEARS.findall(entry.description)
        direct_claim = YEARS.match(entry.description.strip())
        if direct_claim and len(durations) == 1 and not _uncertain_wording(entry.description):
            explicit.append(float(durations[0]))
        try:
            start, end = _date_bound(entry.start_date, True), _date_bound(entry.end_date, False)
        except ValueError:
            start = end = None
        if start and end and end >= start:
            intervals.append((start, end))
    # Never sum explicit duration claims: the entries could overlap.
    lower_bound = max(explicit, default=0.0)
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    # Calendar anniversaries avoid assuming that every year has exactly 365 days.
    complete_years = sum(end.year - start.year - ((end.month, end.day) < (start.month, start.day))
                         for start, end in merged)
    lower_bound = max(lower_bound, complete_years)
    if (explicit or intervals) and lower_bound >= threshold:
        return 'satisfied', (f'Duration-only check: at least {lower_bound:g} years supported; '
                             f'{threshold:g} requested. Role/domain relevance is not verified.')
    if len(candidate.experience) == 1 and len(explicit) == 1 and not intervals:
        if '+' not in candidate.experience[0].description:
            return 'not_satisfied', f'Reported duration {explicit[0]:g} years is below {threshold:g} years.'
    return 'unknown', 'Available experience evidence does not establish the requested duration.'


def _degree(text):
    match = DEGREE.search(text or '')
    if not match:
        return None
    token = re.sub(r'[^a-z]', '', match.group().casefold())
    if token.startswith('bachelor') or token in {'btech', 'be', 'bsc', 'bca'}:
        return 'bachelor'
    if token.startswith('master') or token in {'mtech', 'me', 'msc', 'mca', 'mba'}:
        return 'master'
    if token == 'phd':
        return 'doctorate'
    return None


def _education_field(text):
    """Return (field, safely parsed). A missing parse is not an absent restriction.

    Explicit 'in' fields retain the existing literal comparison. Bare fields cover
    simple discipline phrases such as B.Tech Computer Science, not university,
    grade, accreditation or other qualification constraints.
    """
    degree = DEGREE.search(text)
    if not degree:
        return None, False
    prefix = text[:degree.start()].strip().casefold()
    if prefix not in {'', 'a', 'an', 'required', 'minimum', 'must have', 'requires'}:
        return None, False
    suffix = text[degree.end():].strip().strip('.').strip()
    suffix = re.sub(r'^degree\b', '', suffix, flags=re.I).strip()
    if not suffix:
        return None, True
    if re.fullmatch(r'in\s+[A-Za-z]+(?:[ -][A-Za-z]+){0,5}', suffix, re.I):
        return ' '.join(suffix[3:].casefold().split()), True
    if re.fullmatch(r'(?:[A-Za-z]+[ -]){0,4}(?:science|engineering|technology|mathematics|physics|chemistry|arts|commerce)', suffix, re.I):
        return ' '.join(suffix.casefold().split()), True
    return None, False


def _education(candidate, requirements):
    if not requirements:
        return 'not_required', 'No education requirement supplied.'
    if len(requirements) != 1 or _uncertain_wording(requirements[0]):
        return 'unknown', 'Education wording is optional, ambiguous, or contains multiple requirements.'
    wanted = _degree(requirements[0])
    wanted_field, safely_parsed = _education_field(requirements[0])
    if not safely_parsed:
        return 'unknown', 'Stated education restrictions could not be interpreted safely.'
    if not wanted:
        return 'unknown', 'No supported degree equivalence identified.'
    for entry in candidate.education:
        if _degree(entry.degree) != wanted:
            continue
        source = entry.source_text.splitlines()[0] if entry.source_text else entry.degree or ''
        parsed_field, _ = _education_field(source)
        field = ' '.join((entry.field or parsed_field or '').strip().rstrip('.').casefold().split())
        if wanted_field is None or field == wanted_field:
            return 'satisfied', 'Degree family and stated field match; completion/accreditation is not verified.'
    return 'unknown', 'Education evidence does not establish the requested degree family and field.'


def match_candidate_to_job(candidate_profile: CandidateProfile, job_profile: JobProfile) -> MatchResult:
    """Return a reproducible rule-based score (0–100), not a prediction or decision."""
    candidate, job = candidate_profile, job_profile
    skills, required, preferred = _skills(candidate.skills), _skills(job.required_skills), _skills(job.preferred_skills)
    result = MatchResult(candidate_skills=skills)
    for kind, requested in [('required', required), ('preferred', preferred)]:
        matched = [skill for skill in requested if skill in skills]
        setattr(result, 'matched_' + kind + '_skills', matched)
        setattr(result, 'missing_' + kind + '_skills', [skill for skill in requested if skill not in skills])
        setattr(result, kind + '_skill_match_ratio', len(matched) / len(requested) if requested else None)
    result.experience_match, result.experience_detail = _experience(candidate, job.experience_requirements)
    result.education_match, result.education_detail = _education(candidate, job.education_requirements)
    components = {'required': result.required_skill_match_ratio, 'preferred': result.preferred_skill_match_ratio}
    for name, state in [('experience', result.experience_match), ('education', result.education_match)]:
        components[name] = 1.0 if state == 'satisfied' else 0.0 if state == 'not_satisfied' else None
    result.score_coverage = round(sum(WEIGHTS[name] for name, value in components.items() if value is not None), 10)
    if result.score_coverage:
        result.overall_match_score = round(100 * sum(WEIGHTS[name] * value for name, value in components.items()
                                                     if value is not None) / result.score_coverage, 2)
    matched = result.matched_required_skills + result.matched_preferred_skills
    for key, snippet in candidate.evidence.items():
        if key.startswith('skills.') and snippet:
            for skill in _skills([key[len('skills.'):]]):
                if skill in matched:
                    result.matched_skill_evidence.setdefault(skill, snippet)
    result.explanation = (
        f'Required skills: {len(result.matched_required_skills)}/{len(required)} found. '
        f'Preferred skills: {len(result.matched_preferred_skills)}/{len(preferred)} found. '
        f'Experience: {result.experience_match}. {result.experience_detail} '
        f'Education: {result.education_match}. {result.education_detail} '
        f'Rule-based match score: {result.overall_match_score if result.overall_match_score is not None else "unavailable"}; '
        f'assessed weight coverage: {result.score_coverage:.0%}. '
        'Unknown and unspecified components are excluded, not scored as failures. '
        'Missing skills mean not found in the profile, not proven absent.')
    return result
