"""Simple candidate data; None means unavailable, not a negative assessment."""
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass
class Links:
    linkedin: Optional[str] = None
    github: Optional[str] = None
    portfolio: Optional[str] = None


@dataclass
class Education:
    institution: Optional[str] = None
    degree: Optional[str] = None
    field: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    source_text: str = ""


@dataclass
class Experience:
    company: Optional[str] = None
    role: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: str = ""


@dataclass
class Project:
    name: Optional[str] = None
    description: str = ""
    technologies: List[str] = field(default_factory=list)


@dataclass
class CandidateProfile:
    candidate_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    links: Links = field(default_factory=Links)
    education: List[Education] = field(default_factory=list)
    experience: List[Experience] = field(default_factory=list)
    projects: List[Project] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)
    achievements: List[str] = field(default_factory=list)
    raw_text: str = ""
    # Field paths map to source snippets; these are evidence, not confidence scores.
    evidence: Dict[str, str] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)
