"""
YouTube Shorts agent - Comedy / Absurdist Humor with Strong Hooks, Call-to-Action & Music (60s)
Gemini (script) -> Fish Audio (voice-over) -> Pixabay (footage & music) -> MoviePy (render) -> YouTube (upload)
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

try:  # flush logs line by line so CI shows progress immediately
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

print("DEBUG: Chargement des bibliothèques...")
try:
    from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeAudioClip  # MoviePy 2.x
except ImportError:
    from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeAudioClip  # MoviePy 1.x

import PIL.Image
if not hasattr(PIL.Image, "ANTIALIAS"):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

from google.auth.exceptions import RefreshError
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
print("DEBUG: Bibliothèques chargées avec succès.")


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
GEMINI_MODELS = [m.strip() for m in os.environ.get("GEMINI_MODELS", "").split(",") if m.strip()] or [
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]
GEMINI_TIMEOUT_MS = 120_000

FISH_MODEL = os.environ.get("FISH_MODEL", "s2.1-pro-free").strip()

YT_PRIVACY = os.environ.get("YT_PRIVACY", "public").strip().lower()
if YT_PRIVACY not in ("public", "unlisted", "private"):
    YT_PRIVACY = "public"

OUT_W, OUT_H = 1080, 1920
FALLBACK_QUERIES = ["confused person looking at phone", "person staring blankly", "funny human behavior", "awkward silence"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
class FatalError(RuntimeError):
    """Something retrying can never fix (bad key, no credit, revoked token...)."""


def env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise FatalError(f"Variable d'environnement manquante ou vide : {name}")
    return value


def backoff(attempt: int, base: float = 3.0, cap: float = 60.0) -> float:
    return min(cap, base * 2 ** min(attempt, 10) * random.uniform(0.75, 1.25))


TRANSIENT_HTTP = {408, 425, 429, 500, 502, 503, 504}


def _retry_after(response) -> float:
    try:
        return min(float(response.headers.get("Retry-After", 0)), 120.0)
    except ValueError:
        return 0.0


def request_with_retry(what, method, url, attempts=8, min_bytes=0, fatal_statuses=(401, 402, 403), **kwargs):
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
# 1. Script (Gemini) — 60-Second Comedy Script with Hook & Subscribe CTA
# ---------------------------------------------------------------------------
class Scene(BaseModel):
    narration: str
    video_search_query: str


class VideoScript(BaseModel):
    title: str
    description: str
    tags: list[str]
    scenes: list[Scene]


FATAL_GEMINI = ("api_key_invalid", "api key not valid", "api key expired", "reported as leaked",
                "service_disabled", "unauthenticated")


def suggested_delay(message: str) -> float:
    m = re.search(r"retry(?:Delay)?['\"]?(?:\s+in|:)?\s*['\"]?(\d+(?:\.\d+)?)s", message, re.I)
    return min(float(m.group(1)), 120.0) if m else 0.0


def check_script(script: VideoScript) -> VideoScript:
    script.scenes = [s for s in script.scenes if s.narration.strip() and s.video_search_query.strip()]
    if len(script.scenes) < 5:
        raise ValueError(f"script trop court ({len(script.scenes)} scène(s))")

    title = re.sub(r"[<>]", "", script.title).strip() or "john the blob thoughts"
    if "#shorts" not in title.lower():
        title = f"{title[:90].rstrip()} #shorts"
    script.title = title[:100]
    script.description = re.sub(r"[<>]", "", script.description).strip()

    tags, seen, total = [], set(), 0
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
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=VideoScript,
        ),
    )
    if not response.text:
        raise ValueError("réponse vide")
    return check_script(VideoScript.model_validate_json(response.text))


def generate_script() -> VideoScript:
    print("1. Connexion à Gemini...")
    client = genai.Client(
        api_key=env("GEMINI_API_KEY"),
        http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_MS),
    )
    niches = [
        "absurd shower thoughts that make zero sense",
        "relatable awkward moments everyone pretends didn't happen",
        "terrible life hacks that will ruin your day",
        "roasting everyday human behavior",
        "why being a blob is better than paying taxes"
    ]
    chosen_niche = random.choice(niches)

    prompt = f"""
    You are 'john the blob'—a sarcastic, witty, and chaotic digital observer making fun of human life.
    Create a hilarious, fast-paced, exact **55-to-60-second** YouTube Short comedy script about: {chosen_niche}.

    Structure & Length Requirements:
    - **Massive Hook (Scene 1)**: The very first sentence must be an explosive, shocking, or deeply relatable hook that immediately stops the user from swiping away in the first 3 seconds.
    - **Body (Scenes 2 to 7)**: Build sharp, deadpan humor, ironies of modern life, and comedic observations.
    - **Outro / Call-to-Action (Final Scene)**: End with a witty final joke and a seamless, funny reminder to like and subscribe to john the blob so they don't miss out on more human roasting.
    - **Total Length & Scenes**: Provide exactly **8 to 9 scenes** so the video spans a full minute.

    Provide a viral title (under 70 characters, include #shorts), SEO description, 5 tags, and exactly 8 to 9 scenes.

    For each scene:
    - 'narration': 1-2 punchy, witty sentences with sharp comedic timing.
    - 'video_search_query': 2-3 specific English visual keywords that literally match or represent the narration (e.g., 'confused person looking at phone', 'person staring blankly').
    """

    models = list(GEMINI_MODELS)
    dead = set()
    attempt = 0
    while True:
        live = [m for m in models if m not in dead]
        if not live:
            models = list(GEMINI_MODELS)
            dead.clear()
            live = models

        for model_name in live:
            print(f"Tentative {attempt + 1} avec le modèle {model_name}...")
            try:
                script = ask_gemini(client, model_name, prompt)
                print("Script 60s avec Hook & CTA généré avec succès par Gemini !")
                return script
            except Exception as e:
                msg = str(e)
                if any(k in msg.lower() for k in FATAL_GEMINI):
                    raise FatalError(f"Clé API refusée : {msg[:300]}") from e
                if getattr(e, "code", None) == 404:
                    dead.add(model_name)
                print(f"Échec avec {model_name}: {msg[:300]}")
                time.sleep(1)

        delay = backoff(attempt)
        attempt += 1
        time.sleep(delay)


# ---------------------------------------------------------------------------
# 2. Voice-over (Fish Audio)
# ---------------------------------------------------------------------------
def generate_voiceover(text: str, filename: str):
    print(f"Génération voix off : {text[:30]}...")
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
# 3. Footage & Music (Pixabay)
# ---------------------------------------------------------------------------
def best_mp4_url(videos: dict):
    for size in ("medium", "large", "small", "tiny"):
        url = (videos.get(size) or {}).get("url")
        if url:
            return url
    return None


def download_pixabay_video(query: str, filename: str, used_ids: set):
    print(f"Téléchargement vidéo pour : '{query}'...")
    api_key = env("PIXABAY_API_KEY")
    words = query.split()
    queries = [query] + ([words[0]] if len(words) > 1 else []) + FALLBACK_QUERIES

    for q in queries:
        r = request_with_retry("Pixabay vidéo", "GET", "https://pixabay.com/api/videos/",
                               params={"key": api_key, "q": q[:100], "per_page": 20, "safesearch": "true"})
        try:
            hits = r.json().get("hits", [])
        except ValueError:
            hits = []

        candidates = [h for h in hits if h.get("id") not in used_ids][:5]
        random.shuffle(candidates)
        for hit in candidates:
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
            return
    raise RuntimeError(f"Aucune vidéo Pixabay pour '{query}'")


def download_funny_music(filename: str):
    print("Téléchargement de la musique de fond comique...")
    api_key = env("PIXABAY_API_KEY")
    url = f"https://pixabay.com/api/audio/?key={api_key}&q=funny+comedy+quirky&per_page=10"
    try:
        r = requests.get(url, timeout=15).json()
        hits = r.get("hits", [])
        if hits:
            audio_url = random.choice(hits).get("audio")
            if audio_url:
                res = requests.get(audio_url, timeout=30)
                if len(res.content) > 5000:
                    with open(filename, "wb") as f:
                        f.write(res.content)
                    return True
    except Exception as e:
        print(f"Impossible de récupérer la musique Pixabay Audio : {e}")
    return False


# ---------------------------------------------------------------------------
# 4. Render 60s Video with Music Mixing
# ---------------------------------------------------------------------------
def mp(clip, new_name: str, old_name: str, *args, **kwargs):
    method = getattr(clip, new_name, None) or getattr(clip, old_name)
    return method(*args, **kwargs)


def to_vertical(clip):
    target = OUT_W / OUT_H
    w, h = clip.size
    if w / h > target:
        clip = mp(clip, "cropped", "crop", x_center=w / 2, y_center=h / 2, width=int(h * target), height=h)
    else:
        clip = mp(clip, "cropped", "crop", x_center=w / 2, y_center=h / 2, width=w, height=int(w / target))
    return mp(clip, "resized", "resize", (OUT_W, OUT_H))


def build_video(script: VideoScript, output_path="final_short.mp4"):
    clips, to_close, temp_files, used_ids = [], [], [], set()
    music_path = "background_music.mp3"
    has_music = False

    try:
        if download_funny_music(music_path):
            temp_files.append(music_path)
            has_music = True

        for i, scene in enumerate(script.scenes):
            audio_path = f"scene_{i}.mp3"
            video_path = f"scene_{i}.mp4"
            temp_files += [audio_path, video_path]

            print(f"--- Scène {i+1}/{len(script.scenes)} ---")
            generate_voiceover(scene.narration, audio_path)
            download_pixabay_video(scene.video_search_query, video_path, used_ids)

            raw_audio = AudioFileClip(audio_path)
            raw_video = VideoFileClip(video_path, audio=False)
            to_close += [raw_audio, raw_video]

            end = raw_audio.duration - 0.05 if raw_audio.duration > 0.6 else raw_audio.duration
            audio_clip = mp(raw_audio, "subclipped", "subclip", 0, end)

            video_clip = raw_video
            if video_clip.duration < end:
                video_clip = concatenate_videoclips([video_clip] * (int(end // video_clip.duration) + 1))
            video_clip = mp(video_clip, "subclipped", "subclip", 0, end)
            video_clip = to_vertical(video_clip)
            video_clip = mp(video_clip, "with_audio", "set_audio", audio_clip)
            clips.append(video_clip)

        print("Assemblage de la voix et des vidéos...")
        voice_track = concatenate_videoclips(clips, method="chain")
        to_close.append(voice_track)

        if has_music:
            bg_music = AudioFileClip(music_path)
            to_close.append(bg_music)
            if bg_music.duration < voice_track.duration:
                bg_music = bg_music.loop(duration=voice_track.duration)
            else:
                bg_music = mp(bg_music, "subclipped", "subclip", 0, voice_track.duration)
            
            bg_music = bg_music.volumex(0.15)
            final_audio = CompositeAudioClip([voice_track.audio, bg_music])
            voice_track = mp(voice_track, "with_audio", "set_audio", final_audio)

        print(f"Rendu final de la vidéo (Durée : {voice_track.duration:.1f}s)...")
        voice_track.write_videofile(output_path, fps=24, codec="libx264", audio_codec="aac")
        print("Vidéo 60s prête !")
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
            "description": f"{script.description}\n\n#shorts #comedy #johntheblob",
            "tags": script.tags,
            "categoryId": "23"
        },
        "status": {
            "privacyStatus": YT_PRIVACY,
            "selfDeclaredMadeForKids": False
        }
    }

    media = MediaFileUpload(video_path, chunksize=8 * 1024 * 1024, resumable=True, mimetype="video/mp4")
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    response = request.execute(num_retries=8)
    print(f"Envoyé avec succès ! ID : {response['id']} (https://youtube.com/shorts/{response['id']})")


if __name__ == "__main__":
    print("=== DÉMARRAGE DE L'AGENT 60s (JOHN THE BLOB) ===")
    try:
        script_data = generate_script()
        print(f"Titre choisi : {script_data.title}")

        build_video(script_data)
        upload_to_youtube(script_data)
    except FatalError as e:
        print(f"ERREUR FATALE : {e}")
        sys.exit(1)
