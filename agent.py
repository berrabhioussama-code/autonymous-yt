"""
YouTube Shorts agent — self-learning comedy channel (john the blob)
Gemini (script) -> Fish Audio (voice) -> Pixabay (footage) -> synth music -> MoviePy -> YouTube (scheduled)

Files next to this script (edit them, no code needed):
  prompt.txt          the prompt that makes Gemini write each video
  niches.txt          the daily angles, one per line
  lessons_prompt.txt  the prompt that makes Gemini learn from your results
  memory.txt          long-term memory: your notes, the lessons learned, a log of every video
                      (the workflow commits it after each run)

Publishing: the video is built early, uploaded as PRIVATE with a publish date, and YouTube itself puts it online
at exactly 08:00 and 16:00 Tunisia time. GitHub's late or dropped cron runs no longer matter.

Optional environment variables (everything has a default):
  PUBLISH_TIMES      local publish slots, default "08:00,16:00" ("" = publish immediately)
  PUBLISH_TZ         time zone of the slots, default Africa/Tunis
  GEMINI_MODELS      "model-a,model-b"  override the model order
  MUSIC_VOLUME       0 disables the background music (default 0.10)
  MIN_SCENES / MAX_SCENES   accepted number of scenes (default 5 / 15)
  PIXABAY_PICK_TOP   pick randomly among the N best clips instead of always the best (default 1)
  SCRIPT_ONLY=1      only generate + print the script (no voice, render, upload or memory change)
  NO_UPLOAD=1        build the video but don't upload it (no memory change either)
  YT_PRIVACY         public | unlisted | private (default public; only "public" is scheduled)
  FISH_MODEL, FISH_VOICE_ID, PROMPT_FILE, NICHES_FILE, MEMORY_FILE, LESSONS_PROMPT_FILE,
  ANALYTICS_MIN_AGE_DAYS, SCHEDULE_LEAD_MINUTES, LESSONS_EVERY_HOURS
"""
import os
import re
import sys
import time
import math
import wave
import struct
import random
import hashlib
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from pydantic import BaseModel
from google import genai
from google.genai import types

try:  # flush logs line by line so GitHub Actions shows progress (and retries) immediately
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

print("DEBUG: Chargement des bibliothèques...")
try:
    from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeAudioClip  # MoviePy 2.x
except ImportError:
    from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeAudioClip  # MoviePy 1.x

import PIL.Image
if not hasattr(PIL.Image, "ANTIALIAS"):  # MoviePy 1.x still calls it, Pillow >= 10 removed it
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

from google.auth.exceptions import RefreshError, TransportError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
print("DEBUG: Bibliothèques chargées avec succès.")


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROMPT_FILE = Path(os.environ.get("PROMPT_FILE") or BASE_DIR / "prompt.txt")
NICHES_FILE = Path(os.environ.get("NICHES_FILE") or BASE_DIR / "niches.txt")
LESSONS_PROMPT_FILE = Path(os.environ.get("LESSONS_PROMPT_FILE") or BASE_DIR / "lessons_prompt.txt")
MEMORY_FILE = Path(os.environ.get("MEMORY_FILE") or BASE_DIR / "memory.txt")


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def env_number(name: str, default, cast=float):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return cast(raw)
    except ValueError:
        print(f"ATTENTION : {name}='{raw}' n'est pas un nombre valide, valeur par défaut : {default}")
        return default


# First model that answers wins. gemini-3.8-flash is reported very slow / overloaded since October 2026
# (Google's dev forum), so it comes after the stable 3.6. gemini-2.0 and 1.5 are shut down; 2.5 is restricted
# to accounts that already used it. Put 3.8 first again with GEMINI_MODELS once it recovers.
GEMINI_MODELS = [m.strip() for m in os.environ.get("GEMINI_MODELS", "").split(",") if m.strip()] or [
    "gemini-3.6-flash",
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]
GEMINI_TIMEOUT_MS = 60_000  # a hung request must raise so it can be retried / sent to the next model

FISH_MODEL = os.environ.get("FISH_MODEL", "s2.1-pro-free").strip()

YT_PRIVACY = os.environ.get("YT_PRIVACY", "public").strip().lower()
if YT_PRIVACY not in ("public", "unlisted", "private"):
    YT_PRIVACY = "public"

# When YouTube should put each video online. "" disables scheduling (publish right away, like before).
PUBLISH_TIMES = os.environ.get("PUBLISH_TIMES", "08:00,16:00")
PUBLISH_TZ = os.environ.get("PUBLISH_TZ", "").strip() or "Africa/Tunis"
SCHEDULE_LEAD_MIN = env_number("SCHEDULE_LEAD_MINUTES", 20, int)  # a slot must be at least this far away to be used
SLOT_SPACING_MIN = 45  # a video within 45 minutes of a slot means that slot is already filled

MEMORY_RECENT_VIDEOS = 10  # how many past videos (with all their jokes) Gemini sees in {memory}
LESSONS_EVERY_HOURS = env_number("LESSONS_EVERY_HOURS", 20.0)
MIN_VIDEOS_FOR_LESSONS = 3
LESSONS_GIVE_UP_SEC = 150  # the lessons step is optional: it never holds the video back longer than this

MIN_SCENES = env_number("MIN_SCENES", 5, int)
MAX_SCENES = env_number("MAX_SCENES", 15, int)
MUSIC_VOLUME = env_number("MUSIC_VOLUME", 0.10)
PIXABAY_PICK_TOP = max(1, env_number("PIXABAY_PICK_TOP", 1, int))
MIN_AGE_DAYS = env_number("ANALYTICS_MIN_AGE_DAYS", 3.0)
SCRIPT_ONLY = env_flag("SCRIPT_ONLY")
NO_UPLOAD = env_flag("NO_UPLOAD")
UPLOAD_ENABLED = not (SCRIPT_ONLY or NO_UPLOAD)

OUT_W, OUT_H = 1080, 1920
FALLBACK_QUERIES = ["facepalm", "confused", "laughing", "dog", "cat"]

DEFAULT_NICHES = [  # only used if niches.txt is missing or empty
    "awkward social situations everyone secretly pretends are normal",
    "embarrassing micro-moments we all experience in public",
    "absurd shower thoughts about everyday human habits",
    "ridiculous things humans do at the grocery store or restaurants",
    "weird things people do when they think nobody is watching",
    "unspoken rules of elevators, hallways, and waiting rooms",
    "funny observations about how phones and texting ruined our brains",
    "why adulting and modern daily routines are basically a giant scam",
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
class FatalError(RuntimeError):
    """Something retrying can never fix (bad key, no credit, revoked token, missing file...)."""


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def parse_iso(value):
    """'2026-11-10T07:00:00Z' -> aware datetime; anything unreadable -> None."""
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def env(name: str) -> str:
    """Read a required variable. Strips the stray spaces/newlines that secrets often carry."""
    value = os.environ.get(name, "").strip()
    if not value:
        raise FatalError(f"Variable d'environnement manquante ou vide : {name}")
    return value


def backoff(attempt: int, base: float = 3.0, cap: float = 60.0) -> float:
    """Exponential backoff with jitter: ~3s, 6s, 12s, 24s, 48s, then 60s."""
    return min(cap, base * 2 ** min(attempt, 10) * random.uniform(0.75, 1.25))


TRANSIENT_HTTP = {408, 425, 429, 500, 502, 503, 504}


def _retry_after(response) -> float:
    try:
        return min(float(response.headers.get("Retry-After", 0)), 120.0)
    except ValueError:
        return 0.0


def request_with_retry(what, method, url, attempts=8, min_bytes=0, fatal_statuses=(401, 402, 403), **kwargs):
    """requests.request with a timeout and retries on network errors, 408/429/5xx and truncated bodies."""
    kwargs.setdefault("timeout", (10, 120))
    last = "?"
    for attempt in range(attempts):
        wait = 0.0
        try:
            r = requests.request(method, url, **kwargs)
            if r.status_code in fatal_statuses:
                raise FatalError(f"{what} : HTTP {r.status_code} — {r.text[:300]}")
            if r.status_code < 400 and len(r.content) >= min_bytes:
                return r
            if r.status_code >= 400 and r.status_code not in TRANSIENT_HTTP:
                raise RuntimeError(f"{what} : HTTP {r.status_code} — {r.text[:300]}")
            last = f"HTTP {r.status_code}" if r.status_code >= 400 else "réponse trop courte"
            wait = _retry_after(r)
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError) as e:
            last = type(e).__name__
        if attempt < attempts - 1:
            delay = max(backoff(attempt), wait)
            print(f"{what} : {last} — nouvelle tentative dans {delay:.0f}s ({attempt + 2}/{attempts})")
            time.sleep(delay)
    raise RuntimeError(f"{what} : échec après {attempts} tentatives ({last})")


# ---------------------------------------------------------------------------
# Prompt + niches live in text files (edit them without touching the code)
# ---------------------------------------------------------------------------
PROMPT_PLACEHOLDERS = ("performance_report", "memory", "niche", "seed")


def _read_clean_lines(path: Path) -> list:
    """File lines without the // comment lines. utf-8-sig also swallows the BOM Windows Notepad adds."""
    text = path.read_text(encoding="utf-8-sig")
    return [line for line in text.splitlines() if not line.lstrip().startswith("//")]


def load_prompt_template() -> str:
    if not PROMPT_FILE.is_file():
        raise FatalError(f"Fichier de prompt introuvable : {PROMPT_FILE} — ajoute prompt.txt à côté de agent.py "
                         "(et pense à le commit).")
    template = "\n".join(_read_clean_lines(PROMPT_FILE)).strip()
    if not template:
        raise FatalError(f"{PROMPT_FILE.name} est vide.")
    if "{performance_report}" not in template:
        print("ATTENTION : prompt.txt ne contient pas {performance_report} : Gemini ne verra pas les stats.")
    if "{memory}" not in template:
        print("ATTENTION : prompt.txt ne contient pas {memory} : Gemini ne verra pas la mémoire "
              "(tes notes, les leçons, les dernières vidéos).")
    unknown = sorted(set(re.findall(r"\{([a-z_]+)\}", template)) - set(PROMPT_PLACEHOLDERS))
    if unknown:
        print("ATTENTION : placeholder(s) inconnu(s) dans prompt.txt, laissés tels quels : "
              + ", ".join("{" + u + "}" for u in unknown)
              + " (valides : " + ", ".join("{" + p + "}" for p in PROMPT_PLACEHOLDERS) + ")")
    return template


def render_prompt(template: str, values: dict) -> str:
    """Single pass, so text inside a value (e.g. a video title with braces) is never treated as a placeholder."""
    pattern = r"\{(" + "|".join(map(re.escape, values)) + r")\}"
    return re.sub(pattern, lambda m: str(values[m.group(1)]), template)


def load_niches() -> list:
    niches = []
    if NICHES_FILE.is_file():
        niches = [line.strip() for line in _read_clean_lines(NICHES_FILE) if line.strip()]
        if not niches:
            print(f"{NICHES_FILE.name} est vide : liste intégrée utilisée.")
    else:
        print(f"{NICHES_FILE.name} introuvable : liste intégrée utilisée.")
    return list(dict.fromkeys(niches or DEFAULT_NICHES))


# ---------------------------------------------------------------------------
# Memory (memory.txt): owner notes, lessons learned, log of every video
# ---------------------------------------------------------------------------
MEMORY_HEADER = """// ============================================================================
//  memory.txt — long-term memory of the agent. A normal text file: read it, edit it.
//  Lines starting with // are ignored.
//
//  OWNER NOTES : yours. Read on every run, NEVER modified by the agent. Write rules, taste,
//                things to avoid ("never joke about X", "more dog videos").
//  LESSONS     : rewritten by the agent (about once a day) from your results: what works, what doesn't.
//                Delete or fix any line you disagree with.
//  VIDEO LOG   : one entry added for every video created (title, angle, every joke, YouTube id,
//                publish time) so it never repeats itself.
//  The workflow commits this file after each run, so the memory survives between runs.
// ============================================================================"""
OWNER, LESSONS, LOG = "OWNER NOTES", "LESSONS", "VIDEO LOG"
SECTION_RE = re.compile(r"^===\s*(.+?)\s*===\s*$", re.M)
ENTRY_FIELDS = ("created", "id", "publish", "when", "title", "angle", "hook", "lines")


def _strip_comments(text: str) -> str:
    return "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("//")).strip()


def _one_line(value) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def parse_entries(body: str) -> list:
    """VIDEO LOG body -> list of {field: value}. Tolerant: blank lines, // comments and unknown keys are fine."""
    entries, current = [], None
    for raw in body.splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("---"):
            current = {}
            entries.append(current)
        elif current is not None:
            key, sep, value = line.partition(":")
            if sep:
                current[key.strip().lower()] = value.strip()
    return entries


def render_entry(entry: dict) -> str:
    return "\n".join(["--- video"] + [f"{k}: {_one_line(entry[k])}" for k in ENTRY_FIELDS if entry.get(k)])


@dataclass
class Memory:
    header: str = MEMORY_HEADER
    sections: dict = field(default_factory=dict)  # "OWNER NOTES" / "LESSONS" / "VIDEO LOG" / anything you add

    @property
    def owner_notes(self) -> str:
        return _strip_comments(self.sections.get(OWNER, ""))

    @property
    def lessons(self) -> str:
        return _strip_comments(self.sections.get(LESSONS, ""))

    @property
    def lessons_updated(self):
        m = re.search(r"^//\s*updated:\s*(\S+)", self.sections.get(LESSONS, ""), re.M)
        return parse_iso(m.group(1)) if m else None

    @property
    def entries(self) -> list:
        return parse_entries(self.sections.get(LOG, ""))


def parse_memory(text: str) -> Memory:
    text = text.replace("\r\n", "\n")
    matches = list(SECTION_RE.finditer(text))
    if not matches:  # a plain text file: everything in it is considered owner notes
        return Memory(sections={OWNER: text.strip("\n")})
    sections = {}
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[m.group(1).strip().upper()] = text[m.end():end].strip("\n")
    return Memory(header=text[:matches[0].start()].rstrip("\n") or MEMORY_HEADER, sections=sections)


def render_memory(memory: Memory) -> str:
    names = [OWNER, LESSONS, LOG] + [n for n in memory.sections if n not in (OWNER, LESSONS, LOG)]
    blocks = [memory.header.rstrip("\n")]
    for name in names:
        body = memory.sections.get(name, "").strip("\n")
        blocks.append(f"=== {name} ===" + (f"\n{body}" if body else ""))
    return "\n\n".join(blocks) + "\n"


def load_memory() -> Memory:
    if not MEMORY_FILE.is_file():
        print(f"{MEMORY_FILE.name} introuvable : mémoire vide (le fichier sera créé après la première vidéo).")
        return Memory()
    raw = MEMORY_FILE.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")  # never lose the owner's notes because of an odd editor encoding
    memory = parse_memory(text)
    print(f"Mémoire chargée : {len(memory.entries)} vidéo(s) dans le journal, "
          f"{'des leçons' if memory.lessons else 'pas encore de leçons'}, "
          f"{'des notes du propriétaire' if memory.owner_notes else 'pas de notes du propriétaire'}.")
    return memory


def save_memory(memory: Memory):
    """Atomic write: a crash can never leave a half-written memory.txt."""
    tmp = MEMORY_FILE.with_name(MEMORY_FILE.name + ".tmp")
    tmp.write_text(render_memory(memory), encoding="utf-8")
    os.replace(tmp, MEMORY_FILE)


def memory_for_prompt(memory: Memory) -> str:
    """What Gemini reads in {memory}: the owner's rules, the lessons, and the jokes of the latest videos."""
    parts = []
    if memory.owner_notes:
        parts.append("OWNER NOTES (rules from the channel owner, always follow them):\n" + memory.owner_notes)
    if memory.lessons:
        parts.append("LESSONS LEARNED FROM PAST RESULTS (apply them):\n" + memory.lessons)
    recent = memory.entries[-MEMORY_RECENT_VIDEOS:]
    if recent:
        parts.append("YOUR LAST VIDEOS (never reuse their jokes, premises or hooks):\n" + "\n".join(
            f'- "{e.get("title", "?")}" [angle: {e.get("angle", "?")}] lines: {e.get("lines", "")}' for e in recent))
    return "\n\n".join(parts) or "No memory yet (first videos): nothing to take into account."


def memory_taken(memory: Memory) -> list:
    """Publish times already used by a video listed in the log (works even without YouTube read access)."""
    taken = []
    for e in memory.entries:
        t = parse_iso(e.get("publish", "")) or parse_iso(e.get("created", ""))
        if t:
            taken.append(t)
    return taken


def remember_video(memory: Memory, script, niche: str, video_id: str, publish_at=None):
    entry = {
        "created": utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "id": video_id,
        "publish": publish_at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if publish_at else "immediate",
        "when": f"{publish_at.astimezone(get_publish_tz()):%Y-%m-%d %H:%M} ({PUBLISH_TZ})" if publish_at else "",
        "title": script.title,
        "angle": niche,
        "hook": script.scenes[0].narration,
        "lines": " | ".join(s.narration.replace("|", "/") for s in script.scenes),
    }
    body = memory.sections.get(LOG, "").rstrip()
    memory.sections[LOG] = (body + "\n\n" if body else "") + render_entry(entry)


# ---------------------------------------------------------------------------
# 0. Self-learning analytics (YouTube Data API v3)
# ---------------------------------------------------------------------------
READ_SCOPES = {
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/youtube",
    "https://www.googleapis.com/auth/youtube.force-ssl",
}
SCOPE_HINT = ("Le token YouTube n'a pas le droit de LIRE la chaîne (scope youtube.readonly manquant) : l'analyse "
              "des anciennes vidéos est désactivée. Génère un NOUVEAU refresh token avec les scopes youtube.upload "
              "ET youtube.readonly (un scope ne s'ajoute pas à un token existant), puis mets à jour le secret "
              "YT_REFRESH_TOKEN.")


@dataclass
class Analytics:
    report: str = ("No analytics data available yet (first videos). Ignore this section and rely only on "
                   "today's angle and the rules below.")
    niche_scores: dict = field(default_factory=dict)  # angle -> [views/day of its past videos]
    titles: list = field(default_factory=list)        # every past title: never repeat them
    taken: list = field(default_factory=list)         # publish times already used by a published/scheduled video
    stats: dict = field(default_factory=dict)         # normalised title -> (views/day, like rate %)
    mature_count: int = 0                             # videos old enough to be judged


def niche_label(niche: str) -> str:
    """Short tag stored on each upload, so we can tell later which angle a video used."""
    return "jtb-" + hashlib.sha256(niche.encode("utf-8")).hexdigest()[:8]


def get_youtube_client():
    """Returns (youtube client, set of scopes granted to the token). Fails fast if the token is revoked."""
    creds = Credentials(
        token=None,
        refresh_token=env("YT_REFRESH_TOKEN"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=env("YT_CLIENT_ID"),
        client_secret=env("YT_CLIENT_SECRET")
    )
    for attempt in range(3):
        try:
            creds.refresh(Request())  # checked now, not after minutes of rendering
            break
        except RefreshError as e:
            raise FatalError(
                "YouTube refuse le refresh token (invalid_grant). Si l'écran de consentement OAuth est en mode "
                f"« Testing », les tokens expirent au bout de 7 jours : il faut en régénérer un. Détail : {e}"
            ) from e
        except TransportError:
            if attempt == 2:
                raise
            time.sleep(backoff(attempt))
    granted = set(getattr(creds, "granted_scopes", None) or [])
    return build("youtube", "v3", credentials=creds, cache_discovery=False), granted


def analyse_videos(videos: list, niches: list, now=None) -> Analytics:
    """Turns raw YouTube video resources into the report Gemini reads. Pure function, easy to test."""
    now = now or utcnow()
    label_to_niche = {niche_label(n): n for n in niches}
    titles, rows, taken = [], [], []

    for v in videos:
        sn, st, status = v.get("snippet", {}), v.get("statistics", {}), v.get("status", {})
        title = (sn.get("title") or "").strip()
        if title:
            titles.append(title)
        scheduled_for = parse_iso(status["publishAt"]) if status.get("publishAt") else None
        if scheduled_for:
            taken.append(scheduled_for)  # a video waiting for its slot
        if status.get("privacyStatus") != "public":
            continue  # private videos have no audience: their 0 views would teach Gemini nothing
        published = parse_iso(sn.get("publishedAt"))
        if published is None:
            continue
        taken.append(published)
        age = max((now - published).total_seconds() / 86400, 0.01)
        views = int(st.get("viewCount") or 0)
        likes = int(st["likeCount"]) if st.get("likeCount") is not None else None
        tags = [t.lower() for t in sn.get("tags", [])]
        about = re.sub(r"\s+", " ", re.sub(r"#\w+", "", sn.get("description") or "")).strip()[:110]
        rows.append({
            "title": title, "views": views, "likes": likes, "age": age, "vpd": views / age,
            "like_rate": (100.0 * likes / views) if likes is not None and views else None,
            "niche": next((label_to_niche[t] for t in tags if t in label_to_niche), None),
            "about": about,
        })

    avoid = ""
    if titles:
        avoid = ("ALL PAST TITLES (never repeat them, nor their jokes):\n"
                 + "\n".join(f"- {t}" for t in titles[:40]))

    # A video published yesterday has had no time to perform: judging it would teach Gemini the wrong lesson.
    mature = sorted((r for r in rows if r["age"] >= MIN_AGE_DAYS), key=lambda r: r["vpd"], reverse=True)
    if not mature:
        if not titles:
            return Analytics(taken=taken)
        return Analytics(report=(f"No public video is older than {MIN_AGE_DAYS:g} days yet, so performance can't "
                                 "be judged. Ignore performance and rely on today's angle.\n\n" + avoid),
                         titles=titles, taken=taken)

    def line(rank, r):
        parts = [f'{rank}. "{r["title"]}"',
                 f'{r["views"]:,} views = {r["vpd"]:.0f}/day', f'{r["age"]:.0f} days old']
        if r["like_rate"] is not None:
            parts.append(f'{r["like_rate"]:.1f}% likes')
        if r["niche"]:
            parts.append(f'angle: {r["niche"]}')
        if r["about"]:
            parts.append(f'about: {r["about"]}')
        return " | ".join(parts)

    best = mature[:5]
    worst = mature[max(5, len(mature) - 3):]  # bottom 3, never overlapping the best ones
    scores = {}
    for r in mature:
        if r["niche"]:
            scores.setdefault(r["niche"], []).append(r["vpd"])

    parts = [
        f"{len(mature)} public videos analysed ({len(rows) - len(mature)} newer than {MIN_AGE_DAYS:g} days ignored). "
        f"Median: {statistics.median(r['vpd'] for r in mature):.0f} views/day. "
        "Views are normalised per day so older videos are not unfairly favoured.",
        "BEST PERFORMERS:\n" + "\n".join(line(i, r) for i, r in enumerate(best, 1)),
    ]
    if worst:
        parts.append("WORST PERFORMERS (avoid their pattern):\n" + "\n".join(line(i, r) for i, r in enumerate(worst, 1)))
    if scores:
        ranked = sorted(scores.items(), key=lambda kv: -sum(kv[1]) / len(kv[1]))
        parts.append("ANGLE PERFORMANCE (average views/day):\n" + "\n".join(
            f"- {n}: {sum(v) / len(v):.0f}/day ({len(v)} video{'s' if len(v) > 1 else ''})" for n, v in ranked))
    if avoid:
        parts.append(avoid)
    return Analytics(report="\n\n".join(parts), niche_scores=scores, titles=titles, taken=taken,
                     stats={_norm_title(r["title"]): (r["vpd"], r["like_rate"]) for r in mature},
                     mature_count=len(mature))


def get_channel_performance(niches: list) -> Analytics:
    """Fetches the latest uploads + their stats. Never blocks the pipeline, except for a revoked token."""
    print("0. Analyse des performances des anciennes vidéos sur YouTube...")
    try:
        youtube, granted = get_youtube_client()
        if granted:
            print("Scopes du token : " + ", ".join(sorted(s.rsplit("/", 1)[-1] for s in granted)))
        if granted and not (granted & READ_SCOPES):
            print(SCOPE_HINT)
            return Analytics()

        channels = youtube.channels().list(part="contentDetails", mine=True).execute(num_retries=3)
        items = channels.get("items", [])
        if not items:
            print("Aucune chaîne trouvée pour ce token : analyse ignorée.")
            return Analytics()
        uploads_id = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]

        playlist = youtube.playlistItems().list(
            part="contentDetails", playlistId=uploads_id, maxResults=50).execute(num_retries=3)
        video_ids = [i["contentDetails"]["videoId"] for i in playlist.get("items", [])]
        if not video_ids:
            print("Aucune vidéo publiée pour l'instant.")
            return Analytics()

        videos = youtube.videos().list(
            part="snippet,statistics,status", id=",".join(video_ids)).execute(num_retries=3).get("items", [])
        analytics = analyse_videos(videos, niches)
        print(f"Performances récupérées :\n{analytics.report}")
        return analytics
    except FatalError:
        raise
    except HttpError as e:
        if "insufficient" in str(e).lower():
            print(SCOPE_HINT)
        else:
            print(f"Info: Lecture des stats ignorée ({e}). Poursuite normale.")
    except Exception as e:
        print(f"Info: Lecture des stats ignorée ({type(e).__name__}: {e}). Poursuite normale.")
    return Analytics()


def pick_niche(niches: list, niche_scores: dict):
    """Random until there is enough data, then weighted by how well each angle performed (nothing is ever excluded)."""
    samples = [x for vals in niche_scores.values() for x in vals]
    avg = sum(samples) / len(samples) if samples else 0.0
    if len(samples) < 4 or avg <= 0:
        return random.choice(niches), "aléatoire : pas encore assez de vidéos analysées"
    weights = []
    for n in niches:
        vals = niche_scores.get(n)
        w = sum(vals) / len(vals) if vals else avg  # untested angles get an average chance
        weights.append(max(w, 0.25 * avg))          # a flop never goes below 25% of the average weight
    total = sum(weights)
    print("Poids des angles : " + " | ".join(f"{100 * w / total:.0f}% {n[:35]}" for n, w in zip(niches, weights)))
    return random.choices(niches, weights=weights, k=1)[0], "pondéré par les performances passées"


# ---------------------------------------------------------------------------
# Publishing calendar: which slot is the next one, and is it already filled?
# ---------------------------------------------------------------------------
def parse_publish_times(text: str) -> list:
    """'08:00, 16:00' -> [(8, 0), (16, 0)]. Bad entries are reported and skipped."""
    times = []
    for part in re.split(r"[,\s;]+", (text or "").strip()):
        if not part:
            continue
        m = re.fullmatch(r"(\d{1,2}):(\d{2})", part)
        if m and int(m.group(1)) < 24 and int(m.group(2)) < 60:
            times.append((int(m.group(1)), int(m.group(2))))
        else:
            print(f"ATTENTION : heure de publication ignorée : '{part}' (format attendu : 08:00)")
    return sorted(set(times))


def get_publish_tz():
    try:
        return ZoneInfo(PUBLISH_TZ)
    except Exception:
        if PUBLISH_TZ == "Africa/Tunis":
            return timezone(timedelta(hours=1))  # Tunisia has no daylight saving time
        print(f"ATTENTION : fuseau horaire '{PUBLISH_TZ}' introuvable, UTC utilisé à la place.")
        return timezone.utc


def next_publish_slot(now: datetime, times: list, tz, lead_min=None):
    """First slot that is at least `lead_min` minutes away (time to build + upload), as an aware UTC datetime."""
    lead = timedelta(minutes=SCHEDULE_LEAD_MIN if lead_min is None else lead_min)
    local_now = now.astimezone(tz)
    for day in range(4):
        d = (local_now + timedelta(days=day)).date()
        for hour, minute in times:
            slot = datetime(d.year, d.month, d.day, hour, minute, tzinfo=tz).astimezone(timezone.utc)
            if slot >= now + lead:
                return slot
    return None


def slot_is_taken(slot: datetime, taken: list) -> bool:
    return any(abs((slot - t).total_seconds()) < SLOT_SPACING_MIN * 60 for t in taken)


def plan_publish_slot(analytics: Analytics, memory: Memory):
    """(slot, free). slot is None when scheduling is off. Only the NEXT slot is ever filled: a backup run, a late
    run or a manual run finding it already taken simply does nothing, so videos can never pile up on one slot."""
    times = parse_publish_times(PUBLISH_TIMES)
    if not times:
        return None, True
    slot = next_publish_slot(utcnow(), times, get_publish_tz())
    if slot is None:
        return None, True
    return slot, not slot_is_taken(slot, list(analytics.taken) + memory_taken(memory))


# ---------------------------------------------------------------------------
# 1. Gemini — never gives up on the script
# ---------------------------------------------------------------------------
class Scene(BaseModel):
    narration: str
    video_search_query: str


class VideoScript(BaseModel):
    title: str
    description: str
    tags: list[str]
    scenes: list[Scene]


class Lessons(BaseModel):
    lessons: list[str]


# Errors that no amount of retrying will fix: stop and say why instead of looping forever.
FATAL_GEMINI = ("api_key_invalid", "api key not valid", "api key expired", "reported as leaked",
                "service_disabled", "unauthenticated")


def suggested_delay(message: str) -> float:
    """Gemini 429s say how long to wait ('Please retry in 34s' / retryDelay: '34s')."""
    m = re.search(r"retry(?:Delay)?['\"]?(?:\s+in|:)?\s*['\"]?(\d+(?:\.\d+)?)s", message, re.I)
    return min(float(m.group(1)), 120.0) if m else 0.0


def _norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower().replace("#shorts", "")).strip()


def check_script(script: VideoScript, past_titles=()) -> VideoScript:
    """Repair or reject unusable answers, so a bad answer is retried instead of crashing the render or upload."""
    scenes = [s for s in script.scenes if s.narration.strip() and s.video_search_query.strip()]
    if len(scenes) < MIN_SCENES:
        raise ValueError(f"script trop court ({len(scenes)} scène(s) valides, minimum {MIN_SCENES} : variable MIN_SCENES)")
    script.scenes = scenes[:MAX_SCENES]

    title = re.sub(r"[<>]", "", script.title).strip() or "john the blob thoughts"
    if "#shorts" not in title.lower():
        title = f"{title[:90].rstrip()} #shorts"
    script.title = title[:100]  # YouTube limit
    if _norm_title(script.title) in {_norm_title(t) for t in past_titles}:
        raise ValueError(f"titre déjà utilisé : {script.title}")
    script.description = re.sub(r"[<>]", "", script.description).strip()

    tags, seen, total = [], set(), 0  # YouTube rejects bad tags (<>, more than 500 characters in total)
    for tag in script.tags:
        tag = re.sub(r"[<>#]", "", tag).strip()
        if tag and tag.lower() not in seen and total + len(tag) <= 400:
            tags.append(tag)
            seen.add(tag.lower())
            total += len(tag)
    script.tags = tags
    return script


def discover_gemini_models(client) -> list:
    """Last resort when every configured model is gone: ask the API which Flash models this key can call."""
    skip = ("tts", "image", "live", "audio", "embedding", "robotics", "computer-use", "gemma", "aqa", "transcribe")
    found = []
    try:
        for m in client.models.list():
            name = (m.name or "").removeprefix("models/")
            actions = getattr(m, "supported_actions", None) or []
            if "flash" in name and "generateContent" in actions and not any(s in name for s in skip):
                found.append(name)
    except Exception as e:
        print(f"Impossible de lister les modèles : {e}")

    def best_first(name):  # stable before preview, full before lite, newest version first
        v = re.search(r"gemini-(\d+)(?:\.(\d+))?", name)
        major, minor = (int(v.group(1)), int(v.group(2) or 0)) if v else (0, 0)
        return ("preview" in name or "exp" in name, "lite" in name, -major, -minor)

    return sorted(found, key=best_first)


def make_gemini_client():
    print("1. Connexion à Gemini...")
    return genai.Client(
        api_key=env("GEMINI_API_KEY"),
        http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_MS),
    )


def gemini_json(client, prompt: str, schema, validate=None, give_up_after=None):
    """Ask Gemini for JSON shaped like `schema` (a pydantic model).
    Tries every model, backs off, honours the server's retry hints and NEVER gives up, unless `give_up_after`
    (seconds) is set: then it returns None once that time is spent."""
    started = time.monotonic()
    models = list(GEMINI_MODELS)
    dead = set()  # models that answered 404: they won't come back, so stop asking
    attempt = 0
    while True:  # keeps trying until Gemini answers (Ctrl+C to stop)
        live = [m for m in models if m not in dead]
        if not live:
            print("Aucun des modèles configurés n'existe plus : recherche des modèles disponibles...")
            models = discover_gemini_models(client) or list(GEMINI_MODELS)
            dead.clear()
            live = models

        wait = 0.0
        for model_name in live:
            print(f"Tentative {attempt + 1} avec le modèle {model_name}...")
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    # No temperature/top_p: Gemini 3.x ignores them (and may reject them in future models).
                    # Variety comes from the random seed and the angle in the prompt.
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=schema,
                    ),
                )
                if not response.text:
                    raise ValueError("réponse vide (bloquée ou tronquée)")
                result = schema.model_validate_json(response.text)
                return validate(result) if validate else result
            except Exception as e:
                msg = str(e)
                if any(k in msg.lower() for k in FATAL_GEMINI):
                    raise FatalError(f"Clé API Gemini refusée, inutile de réessayer : {msg[:300]}") from e
                if getattr(e, "code", None) == 404:
                    dead.add(model_name)
                print(f"Échec avec {model_name}: {msg[:300]}")
                wait = max(wait, suggested_delay(msg))
                if give_up_after is not None and time.monotonic() - started >= give_up_after:
                    return None
                time.sleep(1)

        delay = max(backoff(attempt), wait)
        attempt += 1
        if give_up_after is not None and time.monotonic() - started + delay >= give_up_after:
            return None
        print(f"Gemini indisponible pour le moment — nouvelle tentative dans {delay:.0f}s...")
        time.sleep(delay)


def generate_script(template: str, analytics: Analytics, niche: str, memory=None, client=None) -> VideoScript:
    client = client or make_gemini_client()
    memory = memory or Memory()
    prompt = render_prompt(template, {
        "performance_report": analytics.report,
        "memory": memory_for_prompt(memory),
        "niche": niche,
        "seed": random.randint(1000, 999999),
    })
    if SCRIPT_ONLY:
        print("--- PROMPT ENVOYÉ À GEMINI ---\n" + prompt + "\n--- FIN DU PROMPT ---")
    past_titles = list(analytics.titles) + [e.get("title", "") for e in memory.entries]
    script = gemini_json(client, prompt, VideoScript, validate=lambda s: check_script(s, past_titles))
    print("Script optimisé généré avec succès par Gemini !")
    return script


# ---------------------------------------------------------------------------
# 1b. Lessons: Gemini reads the results and rewrites the LESSONS section of memory.txt
# ---------------------------------------------------------------------------
def clean_lessons(result: Lessons) -> Lessons:
    items = []
    for lesson in result.lessons:
        lesson = _one_line(lesson).lstrip("-•*0123456789.) ").strip()
        if lesson:
            items.append(lesson[:240])
    if len(items) < 2:
        raise ValueError("moins de 2 leçons exploitables")
    result.lessons = items[:8]
    return result


def video_log_for_lessons(memory: Memory, analytics: Analytics, limit: int = 15) -> str:
    lines = []
    for e in memory.entries[-limit:]:
        stats = analytics.stats.get(_norm_title(e.get("title", "")))
        if stats:
            result = f"{stats[0]:.0f} views/day" + (f", {stats[1]:.1f}% likes" if stats[1] is not None else "")
        else:
            result = "too new / no data yet"
        lines.append(f'- "{e.get("title", "?")}" | {result} | angle: {e.get("angle", "?")} | lines: {e.get("lines", "")}')
    return "\n".join(lines) or "(no video logged yet)"


def maybe_update_lessons(client, memory: Memory, analytics: Analytics) -> bool:
    """Best effort, about once a day: never holds the video back and never touches the OWNER NOTES."""
    now = utcnow()
    if analytics.mature_count < MIN_VIDEOS_FOR_LESSONS:
        print(f"Leçons : pas assez de vidéos analysées ({analytics.mature_count}/{MIN_VIDEOS_FOR_LESSONS}).")
        return False
    last = memory.lessons_updated
    if last and now - last < timedelta(hours=LESSONS_EVERY_HOURS):
        print("Leçons : déjà mises à jour récemment, on les garde.")
        return False
    if not LESSONS_PROMPT_FILE.is_file():
        print(f"Leçons : {LESSONS_PROMPT_FILE.name} introuvable, étape ignorée.")
        return False
    template = "\n".join(_read_clean_lines(LESSONS_PROMPT_FILE)).strip()
    if not template:
        print(f"Leçons : {LESSONS_PROMPT_FILE.name} est vide, étape ignorée.")
        return False

    prompt = render_prompt(template, {
        "owner_notes": memory.owner_notes or "(none)",
        "previous_lessons": memory.lessons or "(none yet)",
        "performance_report": analytics.report,
        "video_log": video_log_for_lessons(memory, analytics),
    })
    print("Mise à jour des leçons à partir des résultats...")
    result = gemini_json(client, prompt, Lessons, validate=clean_lessons, give_up_after=LESSONS_GIVE_UP_SEC)
    if result is None:
        print("Leçons : Gemini indisponible, on garde les anciennes.")
        return False
    memory.sections[LESSONS] = f"// updated: {now:%Y-%m-%dT%H:%M:%SZ}\n" + "\n".join(f"- {l}" for l in result.lessons)
    save_memory(memory)
    print("Leçons mises à jour :\n" + "\n".join(f"  - {l}" for l in result.lessons))
    return True


# ---------------------------------------------------------------------------
# 2. Voice-over (Fish Audio)
# ---------------------------------------------------------------------------
def generate_voiceover(text: str, filename: str):
    print(f"Voix off : {text}")
    payload = {
        "text": text.strip(),
        "format": "mp3",
        "mp3_bitrate": 128,
        "normalize": True
    }
    voice_id = os.environ.get("FISH_VOICE_ID", "").strip()
    if voice_id:
        payload["reference_id"] = voice_id

    headers = {
        "Authorization": f"Bearer {env('FISH_API_KEY')}",
        "Content-Type": "application/json",
        "model": FISH_MODEL,
    }
    response = request_with_retry("Fish Audio", "POST", "https://api.fish.audio/v1/tts",
                                  min_bytes=1000, json=payload, headers=headers)
    with open(filename, "wb") as f:
        f.write(response.content)


# ---------------------------------------------------------------------------
# 3. Footage (Pixabay)
# ---------------------------------------------------------------------------
def best_mp4_url(videos: dict):
    """Smallest file that still has enough pixels for a 9:16 crop (about 1080p). Pixabay's 'medium' is often only
    1280x720, which would be enlarged 2.7x to fill a vertical frame; 4K is only used if nothing else exists."""
    options = [v for v in (videos.get(s) for s in ("tiny", "small", "medium", "large")) if v and v.get("url")]
    if not options:
        return None
    target = OUT_W / OUT_H

    def enlargement(v):  # how much the centered 9:16 crop must be enlarged to reach 1080x1920
        w, h = v.get("width") or 0, v.get("height") or 0
        if not w or not h:
            return 99.0
        crop_h = h if w / h > target else w / target
        return OUT_H / crop_h

    sharp = [v for v in options if enlargement(v) <= 1.8]
    if sharp:
        return min(sharp, key=lambda v: v.get("size") or 10 ** 18)["url"]
    return min(options, key=enlargement)["url"]


def download_pixabay_video(query: str, filename: str, used_ids: set):
    api_key = env("PIXABAY_API_KEY")
    words = re.sub(r"[^a-zA-Z0-9\s]", " ", query).split()

    # 2-word query first, then the main noun, then generic fallbacks (duplicates removed)
    search_list = []
    if len(words) >= 2:
        search_list.append(" ".join(words[:2]))
    if words:
        search_list += [words[-1], words[0]]
    search_list = list(dict.fromkeys(search_list + FALLBACK_QUERIES))

    for q in search_list:
        print(f"Recherche vidéo Pixabay pour : '{q}'...")
        r = request_with_retry(
            "Pixabay vidéo", "GET", "https://pixabay.com/api/videos/",
            params={
                "key": api_key,
                "q": q[:100],
                "video_type": "film",  # real footage only: animations rarely fit the joke
                "order": "popular",
                "per_page": 15,
                "safesearch": "true"
            }
        )
        try:
            hits = r.json().get("hits", [])
        except ValueError:
            hits = []

        candidates = [h for h in hits if h.get("id") not in used_ids]
        pool = candidates[:PIXABAY_PICK_TOP]  # default 1 = always the best match, no shuffling
        random.shuffle(pool)
        for hit in pool + candidates[PIXABAY_PICK_TOP:PIXABAY_PICK_TOP + 2]:  # + 2 backups if a download fails
            url = best_mp4_url(hit.get("videos", {}))
            if not url:
                continue
            try:
                video = request_with_retry("Pixabay DL", "GET", url, min_bytes=10_000, fatal_statuses=())
            except RuntimeError:
                continue
            with open(filename, "wb") as f:
                f.write(video.content)
            used_ids.add(hit.get("id"))
            print(f"-> Vidéo trouvée avec le mot-clé '{q}' (tags: {hit.get('tags')})")
            return
    raise RuntimeError(f"Aucune vidéo Pixabay pour '{query}'")


# ---------------------------------------------------------------------------
# 4. Silly background music (pure Python synth: no downloads, no MoviePy audio effects)
# ---------------------------------------------------------------------------
def generate_silly_music_wav(filename: str, duration_sec: float, volume: float = 0.10):
    """
    Quirky, bouncy pizzicato/8-bit comedy track written straight to a WAV file.
    Same tune as before, but each video gets a slightly different key and tempo so they don't all sound identical.
    """
    print(f"Génération de la musique de fond comique ({duration_sec:.1f}s)...")
    sample_rate = 44100
    total_samples = int(sample_rate * duration_sec)

    key = 2.0 ** (random.choice([-5, -2, 0, 2, 3, 5, 7]) / 12.0)  # transposition keeps the tune intact
    melody_notes = [f * key for f in (261.63, 329.63, 392.00, 523.25, 392.00, 329.63,
                                      293.66, 349.23, 440.00, 392.00, 329.63, 293.66)]
    bass_notes = [f * key for f in (130.81, 196.00, 146.83, 196.00)]
    step_sec = random.uniform(0.20, 0.25)
    step_samples = int(sample_rate * step_sec)

    with wave.open(filename, "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        chunk = bytearray()
        for i in range(total_samples):
            step_idx = i // step_samples
            pos_in_step = (i % step_samples) / sample_rate

            env_melody = math.exp(-pos_in_step * 14.0)
            freq_m = melody_notes[step_idx % len(melody_notes)]
            freq_m *= (1.0 + 0.03 * math.exp(-pos_in_step * 40.0))

            freq_b = bass_notes[(step_idx // 2) % len(bass_notes)]
            env_bass = math.exp(-pos_in_step * 9.0) if (step_idx % 2 == 0) else 0.0

            melody_val = (math.sin(2.0 * math.pi * freq_m * pos_in_step) +
                          0.4 * math.sin(4.0 * math.pi * freq_m * pos_in_step)) * env_melody
            bass_val = math.sin(2.0 * math.pi * freq_b * pos_in_step) * env_bass * 0.8

            sample = int((melody_val + bass_val) * 0.5 * volume * 32767.0)
            sample = max(-32768, min(32767, sample))
            chunk.extend(struct.pack("<h", sample))

            if len(chunk) >= 65536:
                wav_file.writeframes(chunk)
                chunk.clear()
        if chunk:
            wav_file.writeframes(chunk)
    print("Musique de fond générée avec succès !")


# ---------------------------------------------------------------------------
# 5. Render (MoviePy 1.x and 2.x)
# ---------------------------------------------------------------------------
def mp(clip, new_name: str, old_name: str, *args, **kwargs):
    """Call the MoviePy 2.x method if it exists, otherwise its MoviePy 1.x name (subclip -> subclipped, ...)."""
    method = getattr(clip, new_name, None) or getattr(clip, old_name)
    return method(*args, **kwargs)


def to_vertical(clip):
    """Center-crop to 9:16 first, then resize once: faster and sharper than resizing the whole frame."""
    target = OUT_W / OUT_H
    w, h = clip.size
    if w / h > target:
        clip = mp(clip, "cropped", "crop", x_center=w / 2, y_center=h / 2, width=int(h * target), height=h)
    else:
        clip = mp(clip, "cropped", "crop", x_center=w / 2, y_center=h / 2, width=w, height=int(w / target))
    return mp(clip, "resized", "resize", (OUT_W, OUT_H))


def open_stock_clip(query: str, path: str, used_ids: set, tries: int = 3):
    """Download a clip and make sure it really decodes; if not, take another one instead of crashing at render time."""
    last = "?"
    for attempt in range(1, tries + 1):
        download_pixabay_video(query, path, used_ids)
        clip = None
        try:
            clip = VideoFileClip(path, audio=False)
            clip.get_frame(0)
            if not clip.duration or clip.duration < 0.5:
                raise ValueError("clip trop court")
            return clip
        except Exception as e:
            last = e
            print(f"Clip Pixabay inutilisable ({e}) — j'en prends un autre ({attempt}/{tries})")
            if clip is not None:
                try:
                    clip.close()
                except Exception:
                    pass
    raise RuntimeError(f"Aucun clip Pixabay lisible pour '{query}' ({last})")


def build_video(script: VideoScript, output_path="final_short.mp4"):
    clips, to_close, temp_files, used_ids = [], [], [], set()
    music_path = "background_music.wav"
    temp_files.append(music_path)

    try:
        for i, scene in enumerate(script.scenes):
            audio_path = f"scene_{i}.mp3"
            video_path = f"scene_{i}.mp4"
            temp_files += [audio_path, video_path]

            print(f"--- Scène {i+1}/{len(script.scenes)} ---")
            for attempt in (1, 2):  # an undecodable mp3 is rare, one retry is cheap
                generate_voiceover(scene.narration, audio_path)
                try:
                    raw_audio = AudioFileClip(audio_path)
                    break
                except Exception as e:
                    print(f"Audio illisible ({e})")
                    if attempt == 2:
                        raise
            to_close.append(raw_audio)
            raw_video = open_stock_clip(scene.video_search_query, video_path, used_ids)
            to_close.append(raw_video)

            # mp3 durations are slightly optimistic: trimming the tail avoids read errors on the last samples
            voice_dur = raw_audio.duration - 0.05 if raw_audio.duration > 0.6 else raw_audio.duration
            audio_clip = mp(raw_audio, "subclipped", "subclip", 0, voice_dur)

            # 0.35s pause at the end of each scene so he doesn't yap non-stop
            scene_dur = voice_dur + 0.35

            video_clip = raw_video
            if video_clip.duration < scene_dur:
                video_clip = concatenate_videoclips([video_clip] * (int(scene_dur // video_clip.duration) + 2))
            video_clip = mp(video_clip, "subclipped", "subclip", 0, scene_dur)
            video_clip = to_vertical(video_clip)
            video_clip = mp(video_clip, "with_audio", "set_audio", audio_clip)
            clips.append(video_clip)

        print("Assemblage des scènes...")
        final_video = concatenate_videoclips(clips, method="chain")
        to_close.append(final_video)

        if MUSIC_VOLUME > 0:
            try:
                generate_silly_music_wav(music_path, final_video.duration, volume=MUSIC_VOLUME)
                bg_music = AudioFileClip(music_path)
                to_close.append(bg_music)
                mixed_audio = CompositeAudioClip([final_video.audio, bg_music])
                final_video = mp(final_video, "with_audio", "set_audio", mixed_audio)
            except Exception as e:  # a video without music is better than no video
                print(f"ATTENTION : musique ignorée ({type(e).__name__}: {e})")

        if final_video.duration > 179:
            print(f"ATTENTION : {final_video.duration:.0f}s, au-delà de 3 minutes YouTube ne le traite plus comme un Short.")
        print(f"Rendu final de la vidéo (Durée : {final_video.duration:.1f}s)...")
        final_video.write_videofile(output_path, fps=24, codec="libx264", audio_codec="aac",
                                    preset="veryfast", threads=os.cpu_count() or 2)
        print("Vidéo prête avec musique et rythme comique !")
    finally:
        for c in to_close:
            try:
                c.close()
            except Exception:
                pass
        for path in temp_files:
            if os.path.exists(path):
                os.remove(path)


# ---------------------------------------------------------------------------
# 6. Upload (YouTube) — private + publishAt: YouTube itself publishes at the exact slot
# ---------------------------------------------------------------------------
UPLOAD_ERRORS = {
    "quotaExceeded": "quota de l'API YouTube épuisé pour aujourd'hui, réessaie demain.",
    "uploadLimitExceeded": "limite d'envois du compte atteinte, réessaie plus tard.",
    "insufficientPermissions": "le token n'a pas le scope youtube.upload.",
}


def upload_to_youtube(script: VideoScript, niche: str, publish_at=None, video_path="final_short.mp4"):
    """Returns (video id, the publish time that was scheduled or None if it went out right away)."""
    print("Envoi de la vidéo sur YouTube...")
    youtube, _ = get_youtube_client()

    status = {"privacyStatus": YT_PRIVACY, "selfDeclaredMadeForKids": False}
    scheduled = None
    if publish_at is not None and YT_PRIVACY == "public":
        if publish_at > utcnow() + timedelta(minutes=2):
            scheduled = publish_at.astimezone(timezone.utc)
            status["privacyStatus"] = "private"  # YouTube only accepts publishAt on a private video
            status["publishAt"] = scheduled.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:  # a past publishAt would publish immediately anyway: say it instead of hiding it
            print("ATTENTION : le créneau est passé pendant la création de la vidéo : publication immédiate.")

    body = {
        "snippet": {
            "title": script.title[:100],
            "description": f"{script.description}\n\n#shorts #comedy #johntheblob",
            "tags": script.tags + [niche_label(niche)],  # lets the analytics link this video to its angle
            "categoryId": "23"
        },
        "status": status
    }

    media = MediaFileUpload(video_path, chunksize=8 * 1024 * 1024, resumable=True, mimetype="video/mp4")
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    try:
        response = request.execute(num_retries=8)  # retries chunks on 5xx / rate limits / dropped connections
    except HttpError as e:
        for code, message in UPLOAD_ERRORS.items():
            if code in str(e):
                raise FatalError(f"YouTube : {message}") from e
        raise

    print(f"Envoyé avec succès ! ID : {response['id']} (https://youtube.com/shorts/{response['id']})")
    answered = response.get("status", {})
    if scheduled:
        confirmed = parse_iso(answered.get("publishAt"))
        if answered.get("privacyStatus") == "private" and confirmed and abs((confirmed - scheduled).total_seconds()) < 60:
            local = scheduled.astimezone(get_publish_tz())
            print(f"Programmée : YouTube la publiera tout seul le {local:%Y-%m-%d à %H:%M} ({PUBLISH_TZ}).")
        else:
            print("ATTENTION : YouTube n'a pas confirmé la programmation (publishAt absent de sa réponse) : la vidéo "
                  "risque de rester privée. Un projet API non vérifié force les vidéos en privé.")
    elif answered.get("privacyStatus") and answered["privacyStatus"] != YT_PRIVACY:
        print(f"ATTENTION : demandé '{YT_PRIVACY}' mais YouTube a mis la vidéo en '{answered['privacyStatus']}'. "
              "Un projet API non vérifié force les vidéos en privé.")
    return response["id"], scheduled


def main():
    print("=== DÉMARRAGE DE L'AGENT AUTO-APPRENANT (JOHN THE BLOB) ===")
    try:
        template = load_prompt_template()  # fail fast, before any network call
        niches = load_niches()
        memory = load_memory()

        try:
            analytics = get_channel_performance(niches)
        except FatalError as e:
            if UPLOAD_ENABLED:
                raise
            print(f"Analyse ignorée : {e}")
            analytics = Analytics()

        slot = None
        if UPLOAD_ENABLED:
            slot, free = plan_publish_slot(analytics, memory)
            if slot is not None:
                local = slot.astimezone(get_publish_tz())
                print(f"Prochain créneau de publication : {local:%Y-%m-%d %H:%M} ({PUBLISH_TZ}) = {slot:%H:%M} UTC")
                if not free:
                    print("Ce créneau est déjà rempli par une vidéo programmée ou publiée : rien à créer cette fois.")
                    return

        niche, how = pick_niche(niches, analytics.niche_scores)
        print(f"Angle du jour : {niche} ({how})")

        client = make_gemini_client()
        if UPLOAD_ENABLED:
            try:
                maybe_update_lessons(client, memory, analytics)
            except FatalError:
                raise
            except Exception as e:  # the lessons are a bonus, never a reason to skip a video
                print(f"Leçons ignorées ({type(e).__name__}: {e})")

        script_data = generate_script(template, analytics, niche, memory, client)
        print(f"Titre choisi : {script_data.title}")

        if SCRIPT_ONLY:
            print(script_data.model_dump_json(indent=2))
            print("SCRIPT_ONLY : arrêt avant la voix, le rendu et l'upload.")
            return

        build_video(script_data)
        if NO_UPLOAD:
            print("NO_UPLOAD : vidéo créée (final_short.mp4) mais pas envoyée.")
            return

        video_id, publish_at = upload_to_youtube(script_data, niche, slot)
        try:
            remember_video(memory, script_data, niche, video_id, publish_at)
            save_memory(memory)
            print(f"Mémoire enregistrée dans {MEMORY_FILE.name}.")
        except Exception as e:
            print(f"ATTENTION : mémoire non enregistrée ({type(e).__name__}: {e})")
    except FatalError as e:
        print(f"ERREUR FATALE : {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
