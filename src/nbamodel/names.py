"""Match player names across nba_api, Basketball-Reference and the injury report."""

import re
import unicodedata

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def normalize_name(name: str) -> str:
    """'Dončić, Luka' and 'Luka Doncic' both become 'luka doncic'."""
    if not isinstance(name, str):
        return ""
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    if "," in text:
        last, first = text.split(",", 1)
        text = f"{first} {last}"
    text = re.sub(r"[^a-zA-Z ]", "", text.replace("-", " ")).lower()
    words = [w for w in text.split() if w not in _SUFFIXES]
    return " ".join(words)
