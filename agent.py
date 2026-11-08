"""
YouTube Shorts agent
Gemini (script) -> Fish Audio (voice-over) -> Pixabay (footage) -> MoviePy (render) -> YouTube (upload)
"""
import os
import re
import sys
import time
import random

import requests
from pydantic import BaseModel
from google import genai
from google.genai import types

try:  # flush logs line by line so CI shows progress (and retries) immediately
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

print("DEBUG: Chargement des bibliothèques...")
try:
    from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips  # MoviePy 2.x
except ImportError:
    from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips  # MoviePy 1.x

import PIL.Image
if not hasattr(PIL.Image, "ANTIALIAS"):  # MoviePy 1.x still uses it, Pillow >= 10 removed it
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

from google.auth.exceptions import RefreshError
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
print("DEBUG: Bibliothèques chargées avec succès.")


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
# Newest first. Override without editing the code: GEMINI_MODELS="model-a,model-b"
# (Per Google's docs there is no gemini-3.8-flash-lite text model; the latest Flash-Lite is 3.5.)
GEMINI_MODELS = [m.strip() for m in os.environ.get("GEMINI_MODELS", "").split(",") if m.strip()] or [
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]
GEMINI_TIMEOUT_MS = 120_000  # a hung connection must raise so that it can be retried

FISH_MODEL = os.environ.get("FISH_MODEL", "s2.1-pro-free").strip()

YT_PRIVACY = os.environ.get("YT_PRIVACY", "public").strip().lower()
if YT_PRIVACY not in ("public", "unlisted", "private"):
    YT_PRIVACY = "public"

OUT_W, OUT_H = 1080, 1920
FALLBACK_QUERIES = ["motivation", "gym workout", "running", "sunrise", "city"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
class FatalError(RuntimeError):
    """Something retrying can never fix (bad key, no credit, revoked token...)."""


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


def check_script(script: VideoScript) -> VideoScript:
    """Repair or reject unusable answers, so a bad answer is retried instead of crashing the render/upload."""
    script.scenes = [s for s in script.scenes if s.narration.strip() and s.video_search_query.strip()]
    if len(script.scenes) < 2:
        raise ValueError(f"script inutilisable ({len(script.scenes)} scène(s) valide(s))")

    title = re.sub(r"[<>]", "", script.title).strip() or "Daily motivation"
    if "#shorts" not in title.lower():
        title = f"{title[:90].rstrip()} #shorts"
    script.title = title[:100]  # YouTube limit
    script.description = re.sub(r"[<>]", "", script.description).strip()

    tags, seen, total = [], set(), 0  # YouTube rejects bad tags (<>, > 500 chars in total)
    for tag in script.tags:
        tag = re.sub(r"[<>#]", "", tag).strip()
        if tag and tag.lower() not in seen and total + len(tag) <= 400:
            tags.append(tag)
            seen.add(tag.lower())
            total += len(tag)
    script.tags = tags
    return script


def ask_gemini(client, model_name: str, prompt: str) -> VideoScript:
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        # No temperature/top_p/top_k: Google deprecated them for Gemini 3.x (ignored at best, HTTP 400 on newer models).
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=VideoScript,
        ),
    )
    if not response.text:
        raise ValueError("réponse vide (bloquée ou tronquée)")
    return check_script(VideoScript.model_validate_json(response.text))


def discover_gemini_models(client) -> list:
    """Last resort when every configured model is gone: ask the API which Flash models this key can call."""
    skip = ("tts", "image", "live", "audio", "embedding", "robotics", "computer-use", "gemma", "aqa")
    found = []
    try:
        for m in client.models.list():
            name = (m.name or "").removeprefix("models/")
            actions = getattr(m, "supported_actions", None) or []
            if "flash" in name and "generateContent" in actions and not any(s in name for s in skip):
                found.append(name)
    except Exception as e:
        print(f"Impossible de lister les modèles : {e}")

    def newest_first(name):
        v = re.search(r"gemini-(\d+)(?:\.(\d+))?", name)
        major, minor = (int(v.group(1)), int(v.group(2) or 0)) if v else (0, 0)
        return ("lite" in name, -major, -minor)

    return sorted(found, key=newest_first)


def generate_script() -> VideoScript:
    print("1. Connexion à Gemini...")
    client = genai.Client(
        api_key=env("GEMINI_API_KEY"),
        http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_MS),
    )
    niches = [
        "daily discipline and self-control habits",
        "stoic quotes and mindset shift for success",
        "overcoming laziness and procrastination",
        "how to build unshakeable mental toughness",
        "morning routines of highly successful people"
    ]
    chosen_niche = random.choice(niches)

    prompt = f"""
    Create a viral 30-to-45-second YouTube Short script about {chosen_niche}.
    Provide a catchy title (under 70 characters, include #shorts), a brief SEO description, 5 tags, and 4 to 5 scenes.
    For each scene:
    - 'narration': 1-2 punchy sentences to be spoken aloud.
    - 'video_search_query': 1-2 broad English keywords to find a matching stock video on Pixabay (e.g., 'gym workout motivation', 'dark discipline', 'running').
    """

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
                script = ask_gemini(client, model_name, prompt)
                print("Script généré avec succès par Gemini !")
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
    print(f"Génération de la voix off pour : {text[:30]}...")
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
    for size in ("medium", "large", "small", "tiny"):
        url = (videos.get(size) or {}).get("url")
        if url:
            return url
    return None


def download_pixabay_video(query: str, filename: str, used_ids: set):
    print(f"Téléchargement de la vidéo Pixabay pour la requête : '{query}'...")
    api_key = env("PIXABAY_API_KEY")
    words = query.split()
    # exact query, then its first word, then generic queries, so we never end up with nothing
    queries = [query] + ([words[0]] if len(words) > 1 else []) + FALLBACK_QUERIES

    for q in queries:
        r = request_with_retry("Pixabay (recherche)", "GET", "https://pixabay.com/api/videos/",
                               params={"key": api_key, "q": q[:100], "per_page": 20, "safesearch": "true"})
        try:
            hits = r.json().get("hits", [])
        except ValueError:
            hits = []

        candidates = [h for h in hits if h.get("id") not in used_ids][:5]  # never reuse a clip within a video
        random.shuffle(candidates)
        for hit in candidates:
            url = best_mp4_url(hit.get("videos", {}))
            if not url:
                continue
            try:
                video = request_with_retry("Pixabay (téléchargement)", "GET", url,
                                           min_bytes=10_000, fatal_statuses=())
            except RuntimeError as e:
                print(f"{e} — je tente une autre vidéo")
                continue
            with open(filename, "wb") as f:
                f.write(video.content)
            used_ids.add(hit.get("id"))
            return
    raise RuntimeError(f"Aucune vidéo Pixabay utilisable pour '{query}'")


# ---------------------------------------------------------------------------
# 4. Render (MoviePy 1.x and 2.x)
# ---------------------------------------------------------------------------
def mp(clip, new_name: str, old_name: str, *args, **kwargs):
    """Call the MoviePy 2.x method if it exists, otherwise its MoviePy 1.x name (subclip -> subclipped, ...)."""
    method = getattr(clip, new_name, None) or getattr(clip, old_name)
    return method(*args, **kwargs)


def to_vertical(clip):
    """Center-crop to 9:16 first, then resize once: faster and sharper than resizing the whole frame."""
    target = OUT_W / OUT_H
    w, h = clip.size
    if w / h > target:  # too wide: trim the sides
        clip = mp(clip, "cropped", "crop", x_center=w / 2, y_center=h / 2, width=int(h * target), height=h)
    else:               # too narrow: trim top and bottom
        clip = mp(clip, "cropped", "crop", x_center=w / 2, y_center=h / 2, width=w, height=int(w / target))
    return mp(clip, "resized", "resize", (OUT_W, OUT_H))


def build_video(script: VideoScript, output_path="final_short.mp4"):
    clips, to_close, temp_files, used_ids = [], [], [], set()
    try:
        for i, scene in enumerate(script.scenes):
            audio_path = f"scene_{i}.mp3"
            video_path = f"scene_{i}.mp4"
            temp_files += [audio_path, video_path]

            print(f"--- Traitement de la scène {i+1}/{len(script.scenes)} ---")
            generate_voiceover(scene.narration, audio_path)
            download_pixabay_video(scene.video_search_query, video_path, used_ids)

            raw_audio = AudioFileClip(audio_path)
            raw_video = VideoFileClip(video_path, audio=False)
            to_close += [raw_audio, raw_video]

            # mp3 durations are slightly optimistic; trimming the tail avoids read errors on the last frames
            end = raw_audio.duration - 0.05 if raw_audio.duration > 0.6 else raw_audio.duration
            audio_clip = mp(raw_audio, "subclipped", "subclip", 0, end)

            video_clip = raw_video
            if video_clip.duration < end:  # footage shorter than the narration: repeat it
                video_clip = concatenate_videoclips([video_clip] * (int(end // video_clip.duration) + 1))
            video_clip = mp(video_clip, "subclipped", "subclip", 0, end)
            video_clip = to_vertical(video_clip)
            video_clip = mp(video_clip, "with_audio", "set_audio", audio_clip)
            clips.append(video_clip)

        print("Assemblage final de la vidéo avec MoviePy...")
        final_clip = concatenate_videoclips(clips, method="chain")  # every clip is exactly 1080x1920
        to_close.append(final_clip)
        final_clip.write_videofile(output_path, fps=24, codec="libx264", audio_codec="aac")
        print(f"Vidéo prête ! ({final_clip.duration:.1f}s)")
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
# 5. Upload (YouTube)
# ---------------------------------------------------------------------------
def upload_to_youtube(script: VideoScript, video_path="final_short.mp4"):
    print("Envoi de la vidéo sur YouTube...")
    creds = Credentials(
        token=None,
        refresh_token=env("YT_REFRESH_TOKEN"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=env("YT_CLIENT_ID"),
        client_secret=env("YT_CLIENT_SECRET")
    )
    youtube = build("youtube", "v3", credentials=creds, cache_discovery=False)

    body = {
        "snippet": {
            "title": script.title[:100],
            "description": f"{script.description}\n\n#shorts #motivation",
            "tags": script.tags,
            "categoryId": "27"
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
    except RefreshError as e:
        raise FatalError(
            "YouTube refuse le refresh token (invalid_grant). Si l'écran de consentement OAuth est en mode "
            f"« Testing », les tokens expirent au bout de 7 jours : il faut en régénérer un. Détail : {e}"
        ) from e

    actual = response.get("status", {}).get("privacyStatus")
    print(f"Envoyé avec succès ! ID de la vidéo : {response['id']} (https://youtube.com/shorts/{response['id']})")
    if actual and actual != YT_PRIVACY:
        print(f"ATTENTION : demandé '{YT_PRIVACY}' mais YouTube a mis la vidéo en '{actual}'. "
              "Un projet API non vérifié force les vidéos en privé.")


if __name__ == "__main__":
    print("=== DÉMARRAGE DE L'AGENT YOUTUBE ===")
    try:
        script_data = generate_script()
        print(f"Titre choisi : {script_data.title}")

        build_video(script_data)
        upload_to_youtube(script_data)
    except FatalError as e:
        print(f"ERREUR FATALE : {e}")
        sys.exit(1)
