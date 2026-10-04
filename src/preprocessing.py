"""Stateless model preprocessing; deliberately separate from duplicate keys."""
import re
import unicodedata


def clean_resume(text):
    """Keep skill identities while removing contact details and formatting noise."""
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"(?:https?://|www\.)\S+", " ", text)
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", " ", text)

    # Only long digit sequences qualify; avoid deleting short versions/years.
    def remove_phone(match):
        value = match.group()
        return " " if len(re.sub(r"\D", "", value)) >= 10 else value

    text = re.sub(r"(?<!\w)\+?\d[\d ().-]{7,}\d(?!\w)", remove_phone, text)
    for pattern, replacement in (
        (r"(?<!\w)c\+\+(?!\w)", "cplusplus"),
        (r"(?<!\w)c#(?!\w)", "csharp"),
        (r"(?<!\w)asp\.net\b", "aspnet"),
        (r"(?<!\w)\.net\b", "dotnet"),
        (r"\bnode\.js\b", "nodejs"),
        (r"\breact\.js\b", "reactjs"),
    ):
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())
