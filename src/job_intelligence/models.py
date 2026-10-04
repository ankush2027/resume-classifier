"""Small job-description schema; empty fields represent unavailable information."""
from dataclasses import asdict, dataclass, field
from typing import List, Optional


@dataclass
class JobProfile:
    title: Optional[str] = None
    required_skills: List[str] = field(default_factory=list)
    preferred_skills: List[str] = field(default_factory=list)
    responsibilities: List[str] = field(default_factory=list)
    education_requirements: List[str] = field(default_factory=list)
    experience_requirements: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    raw_text: str = ''

    def to_dict(self):
        return asdict(self)
