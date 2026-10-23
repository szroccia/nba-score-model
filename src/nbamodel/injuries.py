"""Read the official NBA injury report and any manual status overrides.

The NBA posts the report as a PDF several times a day.
"""

import io
import re
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from . import config
from .names import normalize_name

REPORT_URL = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{stamp}.pdf"
STATUSES = ["Out", "Doubtful", "Questionable", "Probable", "Available"]
_PLAYER_RE = re.compile(
    r"(?P<name>[A-Z][\w.'\- ]*?, [A-Z][\w.'\- ]*?)\s+(?P<status>" + "|".join(STATUSES) + r")\b\s*(?P<reason>.*)"
)
# Each game's first row starts with its date and tip-off time, e.g. '10/21/2026 07:30 (ET)'.
_DATE_RE = re.compile(r"(\d{2}/\d{2}/\d{4})\s+\d{1,2}:\d{2}\s*\(ET\)")
NOT_SUBMITTED = "not yet submitted"
_TEAM_ALIASES = {"LA Clippers": "LAC"}


def _team_names() -> dict:
    """Full team name -> tricode, e.g. 'Boston Celtics' -> 'BOS'."""
    from nba_api.stats.static import teams

    names = {t["full_name"]: t["abbreviation"] for t in teams.get_teams()}
    names.update(_TEAM_ALIASES)
    return names


def _candidate_stamps(now_et: datetime):
    """Report timestamps to try, newest first. The NBA has used both
    '2025-01-15_05PM' and '2025-10-21_05_15PM' styles of file name."""
    t = now_et.replace(minute=(now_et.minute // 15) * 15, second=0, microsecond=0)
    for _ in range(4 * 24):
        hour12 = t.strftime("%I")
        ampm = t.strftime("%p")
        day = t.strftime("%Y-%m-%d")
        yield f"{day}_{hour12}_{t.minute:02d}{ampm}"
        if t.minute == 0:
            yield f"{day}_{hour12}{ampm}"
        t -= timedelta(minutes=15)


def latest_report(now: datetime | None = None) -> tuple[str, bytes] | None:
    """Download the most recent injury report PDF from the last 24 hours."""
    now_et = (now or datetime.now(ZoneInfo("America/New_York"))).astimezone(ZoneInfo("America/New_York"))
    for stamp in _candidate_stamps(now_et):
        url = REPORT_URL.format(stamp=stamp)
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                return url, response.read()
        except urllib.error.HTTPError as err:
            if err.code in (403, 404):
                continue
            raise
    return None


def parse_report_text(text: str, team_names: dict) -> pd.DataFrame:
    """Pull (game date, team, player, status, reason) rows out of the report's text.

    Rows look like '10/21/2026 07:30 (ET) LAL@BOS Boston Celtics Tatum, Jayson Out Injury/Illness'.
    A row without a date or team belongs to the last one seen. One report covers more than one
    day of games, so the game date matters. A team that hasn't filed yet gets one row with
    status "not yet submitted" and no player."""
    rows = []
    team = game_date = None
    pattern = re.compile("|".join(re.escape(n) for n in sorted(team_names, key=len, reverse=True)))
    for line in text.splitlines():
        dated = _DATE_RE.search(line)
        if dated:
            game_date = pd.to_datetime(dated.group(1), format="%m/%d/%Y")
        found = pattern.search(line)
        rest = line
        if found:
            team = team_names[found.group(0)]
            rest = line[found.end():]
            if "NOT YET SUBMITTED" in rest.upper():
                rows.append({"GAME_DATE": game_date, "TEAM": team, "PLAYER": "", "STATUS": NOT_SUBMITTED,
                             "REASON": ""})
                continue
        match = _PLAYER_RE.search(rest)
        if team and match:
            rows.append(
                {
                    "GAME_DATE": game_date,
                    "TEAM": team,
                    "PLAYER": match.group("name").strip(),
                    "STATUS": match.group("status").lower(),
                    "REASON": match.group("reason").strip(),
                }
            )
    return pd.DataFrame(rows, columns=["GAME_DATE", "TEAM", "PLAYER", "STATUS", "REASON"])


def for_date(statuses: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """The rows for games on one day. Rows without a date (manual overrides) always apply."""
    dates = pd.to_datetime(statuses["GAME_DATE"]) if "GAME_DATE" in statuses else None
    if dates is None:
        return statuses
    return statuses[dates.isna() | (dates == day.normalize())]


def report_text(pdf_bytes: bytes) -> str:
    import pdfplumber

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        # Default spacing tolerance glues words together in this PDF ("BostonCeltics").
        return "\n".join(page.extract_text(x_tolerance=1) or "" for page in pdf.pages)


def parse_report_pdf(pdf_bytes: bytes) -> pd.DataFrame:
    return parse_report_text(report_text(pdf_bytes), _team_names())


def manual_overrides() -> pd.DataFrame:
    """Your own statuses from manual_status.csv (columns TEAM, PLAYER, STATUS).
    They win over the official report, e.g. for late scratches."""
    path = config.MANUAL_STATUS_PATH
    if not path.exists():
        return pd.DataFrame(columns=["GAME_DATE", "TEAM", "PLAYER", "STATUS", "REASON"])
    df = pd.read_csv(path, comment="#")
    df["STATUS"] = df["STATUS"].str.strip().str.lower()
    df["REASON"] = "manual"
    return df


def play_probabilities(statuses: pd.DataFrame) -> dict:
    """{team tricode: {normalized player name: chance of playing}}. Later rows win."""
    out: dict = {}
    for row in statuses.itertuples():
        prob = config.PLAY_PROBABILITY.get(row.STATUS)
        if prob is None:
            continue
        out.setdefault(row.TEAM, {})[normalize_name(row.PLAYER)] = prob
    return out
