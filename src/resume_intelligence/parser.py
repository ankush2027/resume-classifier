"""Conservative, section-aware rules. Extract mentions, never infer qualifications."""
import json
from pathlib import Path
import re
from urllib.parse import urlsplit
from .models import CandidateProfile, Education, Experience, Project

SECTION_ALIASES = {
    'summary': ['summary', 'professional summary', 'profile', 'about me'],
    'objective': ['objective', 'career objective'],
    'skills': ['skills', 'technical skills', 'key skills', 'core skills', 'technical expertise',
               'programming languages', 'technologies', 'skills and technologies'],
    'education': ['education', 'academic qualifications', 'educational qualifications', 'academic background'],
    'experience': ['experience', 'work experience', 'employment', 'employment history',
                   'professional experience', 'internship', 'internships'],
    'projects': ['projects', 'academic projects', 'personal projects', 'key projects'],
    'certifications': ['certifications', 'certificates', 'licenses and certifications'],
    'achievements': ['achievements', 'awards', 'honors', 'honours', 'awards and achievements'],
    'publications': ['publications', 'research publications'],
    'interests': ['interests', 'hobbies', 'hobbies and interests'],
    'contact': ['contact', 'contact information', 'personal details'],
}
HEADINGS = {alias: key for key, aliases in SECTION_ALIASES.items() for alias in aliases}
TAXONOMY = json.loads(Path(__file__).with_name('skills.json').read_text(encoding='utf-8'))
EMAIL = re.compile(r'(?<![\w.+-])[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b')
URL = re.compile(r'(?<![@\w])(?:https?://[^\s<>]+|www\.[^\s<>]+|(?:[\w-]+\.)?(?:linkedin\.com/in|github\.com)/[^\s<>]+)', re.I)
DEGREE = re.compile(
    r'(?<!\w)(?:B\.?\s?Tech\.?|B\.\s?E\.?|(?-i:BE)|B\.?\s?Sc\.?|BCA|'
    r'M\.?\s?Tech\.?|M\.\s?E\.?|(?-i:ME)|M\.?\s?Sc\.?|MCA|MBA|Ph\.?D\.?|'
    r'Bachelor(?:[’\']s)?(?: of (?:Technology|Engineering|Science|Arts))?|'
    r'Master(?:[’\']s)?(?: of (?:Technology|Engineering|Science|Arts|Business Administration))?)(?!\w)', re.I)
DATE = r'(?:(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+)?(?:19|20)\d{2}|(?:0?[1-9]|1[0-2])/(?:19|20)\d{2}'
DATE_RANGE = re.compile(rf'(?P<start>{DATE})\s*(?:[-–—]|\bto\b)\s*(?P<end>{DATE}|Present|Current|Now)\b', re.I)
ROLE = re.compile(r'\b(?:engineer|developer|analyst|scientist|intern|manager|designer|consultant|architect|lead|researcher|officer|specialist)\b', re.I)


def detect_sections(text):
    """Map standalone or colon-prefixed headings to sections, keeping line structure."""
    sections = {'header': []}
    current = 'header'
    for line in text.splitlines():
        stripped = line.strip().strip('#* ').rstrip(':').strip()
        heading, separator, content = line.strip().partition(':')
        key = HEADINGS.get(stripped.casefold())
        if key is None and separator:
            key = HEADINGS.get(heading.strip().casefold())
        else:
            content = ''
        # Project-local technology lists are evidence for that project, not new sections.
        if current in {'projects', 'experience'} and separator and heading.strip().casefold() == 'technologies':
            key = None
        if key:
            current = key
            sections.setdefault(current, []).append('')
            if content.strip():
                sections[current].append(content.strip())
        else:
            sections.setdefault(current, []).append(line)
    return {key: '\n'.join(lines).strip() for key, lines in sections.items()}


def extract_skills(text, explicit_skills=False):
    """Longest matching aliases win at a position; output follows taxonomy order.

    C/R require uppercase tokens in a skills context. Go requires a skills context
    or the unambiguous alias Golang. A skill mention does not establish proficiency.
    """
    candidates = []
    order = [name for category in TAXONOMY.values() for name in category]
    for category in TAXONOMY.values():
        for name, aliases in category.items():
            for alias in [name] + aliases:
                if name in {'C', 'R'} and not explicit_skills:
                    continue
                if name == 'Go' and alias == 'Go' and not explicit_skills:
                    continue
                flags = 0 if name in {'C', 'R'} else re.I
                # Punctuation boundaries prevent C matching C++, C#, or .NET fragments.
                pattern = r'(?<![\w.+#])' + re.escape(alias) + r'(?![\w+#])'
                for match in re.finditer(pattern, text, flags):
                    candidates.append((match.start(), match.end(), name))
    occupied, found = [], set()
    for start, end, name in sorted(candidates, key=lambda item: (-(item[1]-item[0]), item[0], item[2])):
        if not any(start < right and end > left for left, right in occupied):
            occupied.append((start, end))
            found.add(name)
    return [name for name in order if name in found]


def _name(header):
    """Accept an explicit Name label or a short title-case header near contact info.

    Never guess from the body. Reject headings, role/skill labels, numbers and URLs.
    """
    lines = [line.strip() for line in header.splitlines() if line.strip()][:8]
    has_contact = bool(EMAIL.search(header) or re.search(r'\b(?:phone|mobile|linkedin)\b', header, re.I))
    rejected = set(HEADINGS) | {'resume', 'curriculum vitae', 'cv'}
    for line in lines:
        explicit = re.match(r'^Name\s*:\s*(.+)$', line, re.I)
        value = explicit.group(1).strip() if explicit else line
        if not explicit and not has_contact:
            continue
        words = value.split()
        if (value.casefold() in rejected or not 2 <= len(words) <= 4
                or re.search(r'\b(?:resume|curriculum|vitae|software|development|professional|contact|information|application|candidate)\b', value, re.I)):
            continue
        if ROLE.search(value) or extract_skills(value) or re.search(r'[@\d:/|]', value):
            continue
        if all(re.fullmatch(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*\.?", word, re.UNICODE)
               and (word[0].isupper() or explicit) for word in words):
            return value
    return None


def _dates(text):
    match = DATE_RANGE.search(text)
    if match:
        return match.group('start'), match.group('end')
    graduated = re.search(r'\b(?:graduated|graduation)\s*:?\s*((?:19|20)\d{2})\b', text, re.I)
    return (None, graduated.group(1)) if graduated else (None, None)


def _entries(text, kind):
    """Blank lines or repeated entry headers delimit entries; ambiguous text stays together."""
    entries, lines = [], []
    for line in text.splitlines():
        stripped = line.strip()
        new = False
        if kind == 'education' and DEGREE.search(stripped):
            new = any(DEGREE.search(previous) for previous in lines)
        if kind == 'experience':
            new = bool(ROLE.search(stripped) and (' at ' in stripped or '|' in stripped)
                       and any(ROLE.search(previous) and (' at ' in previous or '|' in previous) for previous in lines))
        if kind == 'projects' and re.match(r'^(?:Project(?: name)?\s*:)', stripped, re.I):
            new = bool(lines)
        if (not stripped or new) and lines:
            entries.append('\n'.join(lines))
            lines = []
        if stripped:
            lines.append(stripped)
    if lines:
        entries.append('\n'.join(lines))
    return entries


def _education(text):
    result = []
    for block in _entries(text, 'education'):
        degree = DEGREE.search(block)
        institution = next((line.strip() for line in block.splitlines()
                            if re.search(r'\b(?:university|institute|college|school|IIT|NIT)\b', line, re.I)), None)
        if not degree and not institution:
            continue
        start, end = _dates(block)
        field = None
        if degree:
            suffix = block[degree.end():].split('\n')[0]
            field_match = re.match(r'\s+(?:in|of)\s+([^|,;]+)', suffix, re.I)
            if field_match:
                field = DATE_RANGE.sub('', field_match.group(1)).strip(' -–—') or None
        # A full mixed line is evidence, not a confidently isolated institution name.
        if institution:
            parts = [part.strip() for part in re.split(r'[|;]', institution)]
            institution = next((part for part in parts if re.search(r'\b(?:university|institute|college|school|IIT|NIT)\b', part, re.I)
                                and not DEGREE.search(part) and not DATE_RANGE.search(part)), None)
        result.append(Education(institution, degree.group().strip() if degree else None,
                                field, start, end, block))
    return result


def _experience(text):
    result = []
    for block in _entries(text, 'experience'):
        company = role = None
        for line in block.splitlines():
            explicit = re.match(r'^(Company|Employer|Role|Title)\s*:\s*(.+)$', line, re.I)
            if explicit:
                if explicit.group(1).lower() in {'company', 'employer'}:
                    company = explicit.group(2).strip()
                else:
                    role = explicit.group(2).strip()
                continue
            parts = re.split(r'\s+at\s+|\s*\|\s*', line)
            if len(parts) >= 2 and ROLE.search(parts[0]) and not line.startswith(('-', '•', '*')):
                # Only parse a short role/company header; bullets stay as description.
                if len(parts[0].split()) <= 6:
                    role = parts[0].strip()
                    company = DATE_RANGE.sub('', parts[1]).strip(' ,()-–—') or None
                    break
        start, end = _dates(block)
        result.append(Experience(company, role, start, end, block))
    return result


def _projects(text):
    result = []
    for block in _entries(text, 'projects'):
        first = block.splitlines()[0]
        explicit = re.match(r'^Project(?: name)?\s*:\s*(.+)$', first, re.I)
        name = explicit.group(1).strip() if explicit else None
        if name is None and len(block.splitlines()) > 1 and len(first.split()) <= 8:
            if not first.startswith(('-', '•', '*')) and not re.match(r'^(built|developed|implemented|technologies|tech stack|description)\b', first, re.I):
                name = first.rstrip(':')
        technologies = extract_skills(block)
        for line in block.splitlines():
            if re.match(r'^(?:Technologies|Tech stack|Languages)\s*:', line, re.I):
                technologies += extract_skills(line, explicit_skills=True)
        result.append(Project(name, block, list(dict.fromkeys(technologies))))
    return result


def _list_entries(text):
    return [line.strip().lstrip('-•* ').strip() for line in text.splitlines() if line.strip()]


def parse_resume(text) -> CandidateProfile:
    """Parse text locally without altering the source. Non-text/None yields an empty profile."""
    if not isinstance(text, str):
        return CandidateProfile()
    profile = CandidateProfile(raw_text=text)
    if not text.strip():
        return profile
    # Keep source untouched; normalize CRLF only in the working copy.
    sections = detect_sections(text.replace('\r\n', '\n').replace('\r', '\n'))
    header = sections.get('header', '')
    contact = header + '\n' + sections.get('contact', '')
    profile.candidate_name = _name(contact)
    email = EMAIL.search(text)
    if email:
        profile.email = email.group()
    for match in re.finditer(r'(?<!\w)\+?\(?\d[\d ().-]{7,}\d(?!\w)', contact):
        if 10 <= len(re.sub(r'\D', '', match.group())) <= 15:
            if not DATE_RANGE.search(match.group()):
                profile.phone = match.group().strip()
                break
    location = re.search(r'^(?:Location|Address|Based in)\s*:\s*(.+)$', contact, re.I | re.M)
    if location:
        profile.location = location.group(1).strip()
    for match in URL.finditer(contact):
        value = match.group().rstrip('.,;)]}')
        url = value if re.match(r'https?://', value, re.I) else 'https://' + value
        try:
            parsed = urlsplit(url)
            host = (parsed.hostname or '').lower()
        except ValueError:
            continue  # Invalid URL syntax is not contact evidence.
        if not host or '.' not in host:
            continue
        if host == 'linkedin.com' or host.endswith('.linkedin.com'):
            field = 'linkedin'
        elif host in {'github.com', 'www.github.com'}:
            field = 'github'
        else:
            field = 'portfolio'
        if getattr(profile.links, field) is None:
            setattr(profile.links, field, url)
            profile.evidence['links.' + field] = value
    # Bare personal domains are accepted only with an explicit website/portfolio label.
    if profile.links.portfolio is None:
        match = re.search(r'^(?:Portfolio|Website)\s*:\s*((?:[\w-]+\.)+[a-z]{2,}(?:/\S*)?)', contact, re.I | re.M)
        if match:
            profile.links.portfolio = 'https://' + match.group(1).rstrip('.,;')
            profile.evidence['links.portfolio'] = match.group(0)
    for field in ['candidate_name', 'email', 'phone', 'location']:
        value = getattr(profile, field)
        if value:
            profile.evidence[field] = next((line for line in text.splitlines() if value in line), value)
    profile.education = _education(sections.get('education', ''))
    profile.experience = _experience(sections.get('experience', ''))
    profile.projects = _projects(sections.get('projects', ''))
    profile.certifications = _list_entries(sections.get('certifications', ''))
    profile.achievements = _list_entries(sections.get('achievements', ''))
    for section, content in sections.items():
        if section not in {'header', 'contact', 'skills', 'summary', 'objective', 'experience', 'projects'}:
            continue
        for skill in extract_skills(content, explicit_skills=section == 'skills'):
            if skill not in profile.skills:
                profile.skills.append(skill)
                snippet = next((line for line in content.splitlines()
                                if skill in extract_skills(line, explicit_skills=section == 'skills')), content)
                profile.evidence['skills.' + skill] = section + ': ' + snippet
    for project in profile.projects:
        for skill in project.technologies:
            if skill not in profile.skills:
                profile.skills.append(skill)
                profile.evidence['skills.' + skill] = 'projects: ' + project.description
    for field in ['education', 'experience', 'projects', 'certifications', 'achievements']:
        if getattr(profile, field):
            profile.evidence[field] = sections.get(field, '')
    return profile
