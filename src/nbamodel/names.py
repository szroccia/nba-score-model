"""Match player names across nba_api, Basketball-Reference and the injury report."""

import re
import unicodedata

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
# Players who go by a different first name in one source (after the cleanup below).
_ALIASES = {"ron holland": "ronald holland"}


def normalize_name(name: str) -> str:
    """'Dončić, Luka' and 'Luka Doncic' both become 'luka doncic'."""
    if not isinstance(name, str):
        return ""
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    if "," in text:
        last, first = text.split(",", 1)
        text = f"{first} {last}"
    text = re.sub(r"[^a-zA-Z ]", "", text.replace("-", " ")).lower()
    key = " ".join(w for w in text.split() if w not in _SUFFIXES)
    return _ALIASES.get(key, key)
