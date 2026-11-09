"""
YouTube Shorts agent — self-learning comedy channel (john the blob)
Gemini (script) -> Fish Audio (voice) -> Pixabay (footage) -> synth music -> MoviePy -> YouTube

Files next to this script (edit them, no code needed):
  prompt.txt   the prompt sent to Gemini
  niches.txt   the daily angles, one per line

Optional environment variables (everything has a default):
  GEMINI_MODELS      "model-a,model-b"  override the model order
  MUSIC_VOLUME       0 disables the background music (default 0.10)
  MIN_SCENES / MAX_SCENES   accepted number of scenes (default 5 / 15)
  PIXABAY_PICK_TOP   pick randomly among the N best clips instead of always the best (default 1)
  SCRIPT_ONLY=1      only generate + print the script (no voice, render or upload): to test prompt.txt
  NO_UPLOAD=1        build the video but don't upload it
  YT_PRIVACY         public | unlisted | private (default public)
  FISH_MODEL, FISH_VOICE_ID, PROMPT_FILE, NICHES_FILE, ANALYTICS_MIN_AGE_DAYS
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
from datetime import datetime, timezone
from pathlib import Path

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
PROMPT_PLACEHOLDERS = ("performance_report", "niche", "seed")


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


def niche_label(niche: str) -> str:
    """Short invisible-ish tag stored on each upload, so we can tell later which angle a video used."""
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
    now = now or datetime.now(timezone.utc)
    label_to_niche = {niche_label(n): n for n in niches}
    titles, rows = [], []

    for v in videos:
        sn, st, status = v.get("snippet", {}), v.get("statistics", {}), v.get("status", {})
        title = (sn.get("title") or "").strip()
        if title:
            titles.append(title)
        if status.get("privacyStatus") != "public":
            continue  # private videos have no audience: their 0 views would teach Gemini nothing
        try:
            published = datetime.fromisoformat(sn["publishedAt"].replace("Z", "+00:00"))
        except (KeyError, ValueError):
            continue
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
            return Analytics()
        return Analytics(report=(f"No public video is older than {MIN_AGE_DAYS:g} days yet, so performance can't "
                                 "be judged. Ignore performance and rely on today's angle.\n\n" + avoid),
                         titles=titles)

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
    return Analytics(report="\n\n".join(parts), niche_scores=scores, titles=titles)


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
# 1. Script (Gemini) — never gives up
# ---------------------------------------------------------------------------
class Scene(BaseModel):
    narration: str
    video_search_query: str


class VideoScript(BaseModel):
    title: str
    description: str
    tags: list[str]
    scenes: list[Scene]


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


def ask_gemini(client, model_name: str, prompt: str, past_titles=()) -> VideoScript:
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        # No temperature/top_p: Gemini 3.x ignores them (and may reject them in future models).
        # Variety comes from the random seed and the angle in the prompt.
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=VideoScript,
        ),
    )
    if not response.text:
        raise ValueError("réponse vide (bloquée ou tronquée)")
    return check_script(VideoScript.model_validate_json(response.text), past_titles)


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


def generate_script(template: str, analytics: Analytics, niche: str) -> VideoScript:
    print("1. Connexion à Gemini...")
    client = genai.Client(
        api_key=env("GEMINI_API_KEY"),
        http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_MS),
    )
    prompt = render_prompt(template, {
        "performance_report": analytics.report,
        "niche": niche,
        "seed": random.randint(1000, 999999),
    })
    if SCRIPT_ONLY:
        print("--- PROMPT ENVOYÉ À GEMINI ---\n" + prompt + "\n--- FIN DU PROMPT ---")

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
                script = ask_gemini(client, model_name, prompt, analytics.titles)
                print("Script optimisé généré avec succès par Gemini !")
                return script
            except Exception as e:
                msg = str(e)
                if any(k in msg.lower() for k in FATAL_GEMINI):
                    raise FatalError(f"Clé API Gemini refusée, inutile de réessayer : {msg[:300]}") from e
                if getattr(e, "code", None) == 404:
                    dead.add(model_name)
                print(f"Échec avec {model_name}: {msg[:300]}")
                wait = max(wait, suggested_delay(msg))
                time.sleep(1)

        delay = max(backoff(attempt), wait)
        attempt += 1
        print(f"Gemini indisponible pour le moment — nouvelle tentative dans {delay:.0f}s...")
        time.sleep(delay)


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
# 6. Upload (YouTube)
# ---------------------------------------------------------------------------
UPLOAD_ERRORS = {
    "quotaExceeded": "quota de l'API YouTube épuisé pour aujourd'hui, réessaie demain.",
    "uploadLimitExceeded": "limite d'envois du compte atteinte, réessaie plus tard.",
    "insufficientPermissions": "le token n'a pas le scope youtube.upload.",
}


def upload_to_youtube(script: VideoScript, niche: str, video_path="final_short.mp4"):
    print("Envoi de la vidéo sur YouTube...")
    youtube, _ = get_youtube_client()

    body = {
        "snippet": {
            "title": script.title[:100],
            "description": f"{script.description}\n\n#shorts #comedy #johntheblob",
            "tags": script.tags + [niche_label(niche)],  # lets the analytics link this video to its angle
            "categoryId": "23"
        },
        "status": {
            "privacyStatus": YT_PRIVACY,
            "selfDeclaredMadeForKids": False
        }
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
    actual = response.get("status", {}).get("privacyStatus")
    if actual and actual != YT_PRIVACY:
        print(f"ATTENTION : demandé '{YT_PRIVACY}' mais YouTube a mis la vidéo en '{actual}'. "
              "Un projet API non vérifié force les vidéos en privé.")


def main():
    print("=== DÉMARRAGE DE L'AGENT AUTO-APPRENANT (JOHN THE BLOB) ===")
    try:
        template = load_prompt_template()  # fail fast, before any network call
        niches = load_niches()

        try:
            analytics = get_channel_performance(niches)
        except FatalError as e:
            if UPLOAD_ENABLED:
                raise
            print(f"Analyse ignorée : {e}")
            analytics = Analytics()

        niche, how = pick_niche(niches, analytics.niche_scores)
        print(f"Angle du jour : {niche} ({how})")
        script_data = generate_script(template, analytics, niche)
        print(f"Titre choisi : {script_data.title}")

        if SCRIPT_ONLY:
            print(script_data.model_dump_json(indent=2))
            print("SCRIPT_ONLY : arrêt avant la voix, le rendu et l'upload.")
            return

        build_video(script_data)
        if NO_UPLOAD:
            print("NO_UPLOAD : vidéo créée (final_short.mp4) mais pas envoyée.")
            return
        upload_to_youtube(script_data, niche)
    except FatalError as e:
        print(f"ERREUR FATALE : {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
