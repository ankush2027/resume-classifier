"""Preserve snippets from candidate fields; do not turn generated summaries into evidence."""
import re
from src.resume_intelligence.parser import detect_sections
from .models import Evidence
from .rules import classify, mentioned, related, MULTIPLIERS


def snippets(text):
    # Split clauses while retaining original text, punctuation and technical dots.
    for line in text.splitlines():
        for fragment in re.split(r'(?<=[.!?;])\s+|\s+but\s+', line):
            fragment = fragment.strip()
            if fragment:
                yield fragment


def candidate_sources(candidate):
    """Each tuple identifies the actual source field, not an invented citation."""
    sources = []
    for field, kind, attribute in [('experience', 'experience', 'description'),
                                    ('projects', 'project', 'description'),
                                    ('education', 'education', 'source_text')]:
        for index, entry in enumerate(getattr(candidate, field)):
            text = getattr(entry, attribute)
            if text:
                sources.append((kind, field, f'{field}[{index}].{attribute}', text))
            if field == 'projects':
                for number, technology in enumerate(entry.technologies):
                    sources.append(('project', field, f'{field}[{index}].technologies[{number}]', technology))
    for field, kind in [('certifications', 'certification'), ('achievements', 'achievement')]:
        for index, text in enumerate(getattr(candidate, field)):
            sources.append((kind, field, f'{field}[{index}]', text))
    # Raw sections supply header/summary/skills and any material missed by the parser.
    types = {'skills': 'skills', 'projects': 'project', 'experience': 'experience',
             'education': 'education', 'certifications': 'certification',
             'achievements': 'achievement', 'summary': 'summary', 'objective': 'summary'}
    for section, text in detect_sections(candidate.raw_text).items():
        sources.append((types.get(section, 'other'), section, 'raw_text', text))
    return sources


def extract_evidence(requirement, sources):
    """One record per concept/snippet; keep the strongest source, never sum repeats."""
    records = {}
    for kind, section, path, text in sources:
        for snippet in snippets(text):
            for concept in [requirement] + related(requirement):
                if not mentioned(concept, snippet):
                    continue
                strength, supports, reason = classify(kind, snippet)
                item = Evidence(requirement, concept, kind, snippet, strength, section,
                                'direct' if concept == requirement else 'transferable',
                                path, supports, reason)
                key = (concept, ' '.join(snippet.casefold().split()))
                previous = records.get(key)
                if (previous is None or MULTIPLIERS[strength] > MULTIPLIERS[previous.strength]
                        or (strength == previous.strength and kind == 'skills' and previous.source_type != 'skills')):
                    records[key] = item
    return sorted(records.values(), key=lambda item: (
        item.relevance != 'direct', not item.supports_requirement,
        -MULTIPLIERS[item.strength], item.source_type, item.source_text, item.concept))
