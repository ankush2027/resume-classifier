"""Small auditable rules; frequency never contributes to strength."""
import re
from src.resume_intelligence.parser import extract_skills

MULTIPLIERS = {'weak': 0.3, 'moderate': 0.7, 'strong': 1.0}
# Related technologies are deliberately not added as aliases to the skill taxonomy.
TRANSFER_GROUPS = (
    ('FastAPI', 'Flask', 'Django', 'Spring', 'REST API'),
    ('PostgreSQL', 'MySQL', 'SQL', 'MongoDB'),
    ('AWS', 'Azure', 'GCP'),
)
DOMAINS = {
    'backend': r'\b(?:backend|back-end|REST\s+APIs?|APIs?|server-side)\b',
    'frontend': r'\b(?:frontend|front-end|user interface|UI)\b',
    'data analysis': r'\b(?:data analysis|analytics|notebooks?)\b',
    'cloud': r'\b(?:cloud|infrastructure)\b',
}
ACTION = re.compile(r'\b(?:built|developed|implemented|maintained|deployed|containerized|designed|created|automated|optimized|integrated|tested|operated|migrated|managed|worked on)\b', re.I)
OBJECT = re.compile(r'\b(?:APIs?|services?|applications?|systems?|pipelines?|models?|databases?|backend|frontend|infrastructure|platforms?|notebooks?|analysis|containers?|websites?|tools?)\b', re.I)
UNSUPPORTED = re.compile(r"\b(?:no|not|never|without|lack|lacking|plan(?:ning)?|want|wish|interest|interested|unrelated|aspir(?:e|ing)|learning|hoping|would|could|will)\b|\b(?:don[’']t|didn[’']t|haven[’']t)\b", re.I)
INDIRECT = re.compile(r'\b(?:team where|team used|team uses|exposure to|familiar with|aware of|observed|assisted|helped)\b', re.I)
CERTIFICATION = re.compile(r'\b(?:certified|certification|certificate|completed|credential)\b', re.I)


def mentioned(concept, text):
    """Reuse canonical aliases; unlisted job concepts require literal boundaries."""
    if concept in extract_skills(text, explicit_skills=True):
        return True
    # Literal fallback supports job concepts such as JWT/authentication without inference.
    return bool(re.search(r'(?<![\w.+#])' + re.escape(concept) + r'(?![\w+#])', text, re.I))


def classify(source_type, text):
    if (UNSUPPORTED.search(text) or
            re.search(r'\b(?:familiar with|knowledge only|only knowledge)\b', text, re.I)):
        return 'weak', False, 'Negated, hypothetical, interest-only, or familiarity wording does not establish use.'
    if INDIRECT.search(text):
        return 'weak', True, 'Indirect exposure wording does not establish personal responsibility.'
    concrete = bool(ACTION.search(text) and OBJECT.search(text))
    if source_type == 'experience' and concrete:
        return 'strong', True, 'Concrete work action and task are described with the concept.'
    if source_type in {'project', 'education', 'achievement'} and concrete:
        return 'moderate', True, 'Concrete non-employment use is described.'
    if source_type == 'certification' and CERTIFICATION.search(text):
        return 'moderate', True, 'Credential mention is supporting evidence, not professional experience.'
    return 'weak', True, 'Mention lacks a concrete task or independently attributable work context.'


def related(concept):
    return [item for group in TRANSFER_GROUPS if concept in group for item in group if item != concept]


def domains(text):
    return [name for name, pattern in DOMAINS.items() if re.search(pattern, text, re.I)]
