"""
YouTube Shorts agent - Comedy / Absurdist Humor (john the blob)
Fixes: Guaranteed built-in silly music, accurate 1-2 word visual search, punchy pacing (less yapping)
"""
import os
import re
import sys
import time
import math
import wave
import struct
import random

import requests
from pydantic import BaseModel
from google import genai
from google.genai import types

try:
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
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]
GEMINI_TIMEOUT_MS = 120_000

FISH_MODEL = os.environ.get("FISH_MODEL", "s2.1-pro-free").strip()

YT_PRIVACY = os.environ.get("YT_PRIVACY", "public").strip().lower()
if YT_PRIVACY not in ("public", "unlisted", "private"):
    YT_PRIVACY = "public"

OUT_W, OUT_H = 1080, 1920
FALLBACK_QUERIES = ["facepalm", "confused", "laughing", "dog", "cat"]


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
# 1. Script (Gemini) — Less Yapping, Better Visual Match
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
            temperature=1.0,
            top_p=0.95,
        ),
    )
    if not response.text:
        raise ValueError("réponse vide")
    return check_script(VideoScript.model_validate_json(response.text))


def discover_gemini_models(client) -> list:
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
    return found


def generate_script() -> VideoScript:
    print("1. Connexion à Gemini...")
    client = genai.Client(
        api_key=env("GEMINI_API_KEY"),
        http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_MS),
    )
    niches = [
        "absurd shower thoughts about everyday objects",
        "awkward social situations everyone secretly hates",
        "ridiculous things humans do at the grocery store",
        "roasting human morning routines and coffee addiction",
        "weird things people do when they think nobody is watching",
        "unspoken rules of elevators and waiting rooms",
        "funny observations about pets ruling human households",
        "why adulting is basically a giant scam"
    ]
    chosen_niche = random.choice(niches)
    random_seed_trigger = f"Random seed: {random.randint(1000, 999999)}"

    prompt = f"""
    {random_seed_trigger}
    You are 'john the blob'—a deadpan, witty digital observer roasting human behavior.
    Create a punchy YouTube Short comedy script about: {chosen_niche}.

    CRITICAL RULES TO AVOID "YAPPING" AND FIX STOCK FOOTAGE MATCHING:
    1. **NO YAPPING**: Each scene's 'narration' MUST be **1 single short, punchy sentence (maximum 10 to 13 words)**. No long-winded rants or complex filler words. Let the joke hit fast!
    2. **EXACTLY 10 SCENES**: Provide 10 rapid-fire scenes so the video moves quickly with visual variety.
    3. **Hook (Scene 1)**: A short, hilarious 1-sentence hook that grabs attention in 3 seconds.
    4. **Outro (Scene 10)**: A quick final punchline telling them to like and subscribe to john the blob.
    5. **SIMPLE 1-2 WORD SEARCH QUERIES**: Pixabay fails when given long phrases! For 'video_search_query', provide ONLY **1 or 2 ultra-common concrete English nouns/actions** that literally appear in your sentence and are guaranteed to exist on a stock video site (e.g., 'drinking coffee', 'microwave', 'sleeping bed', 'supermarket', 'typing laptop', 'alarm clock', 'traffic jam'). NEVER use 3+ words or abstract concepts.

    Provide a viral title (under 70 characters, include #shorts), SEO description, 5 tags, and 10 scenes.
    """

    models = discover_gemini_models(client) or list(GEMINI_MODELS)
    attempt = 0
    while True:
        for model_name in models:
            print(f"Tentative {attempt + 1} avec le modèle {model_name}...")
            try:
                script = ask_gemini(client, model_name, prompt)
                print("Script généré avec succès par Gemini !")
                return script
            except Exception as e:
                msg = str(e)
                if any(k in msg.lower() for k in FATAL_GEMINI):
                    raise FatalError(f"Clé API refusée : {msg[:300]}") from e
                print(f"Échec avec {model_name}: {msg[:300]}")
                time.sleep(1)

        delay = backoff(attempt)
        attempt += 1
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
# 3. Footage (Pixabay — Exact Relevance Sorting, No Random Shuffling)
# ---------------------------------------------------------------------------
def best_mp4_url(videos: dict):
    for size in ("medium", "large", "small", "tiny"):
        url = (videos.get(size) or {}).get("url")
        if url:
            return url
    return None


def download_pixabay_video(query: str, filename: str, used_ids: set):
    api_key = env("PIXABAY_API_KEY")
    clean_q = re.sub(r"[^a-zA-Z0-9\s]", "", query).strip()
    words = clean_q.split()

    # Try the 2-word query first, then the main noun, then fallbacks
    search_list = []
    if len(words) >= 2:
        search_list.append(f"{words[0]} {words[1]}")
    if words:
        search_list.append(words[-1])  # Usually the main noun in English (e.g. "hot coffee" -> "coffee")
        search_list.append(words[0])
    search_list.extend(FALLBACK_QUERIES)

    for q in search_list:
        print(f"Recherche vidéo Pixabay pour : '{q}'...")
        r = request_with_retry(
            "Pixabay vidéo", "GET", "https://pixabay.com/api/videos/",
            params={
                "key": api_key,
                "q": q[:50],
                "order": "popular",
                "per_page": 15,
                "safesearch": "true"
            }
        )
        try:
            hits = r.json().get("hits", [])
        except ValueError:
            hits = []

        # Pick the TOP matching unused clip directly — DO NOT shuffle so it stays relevant!
        candidates = [h for h in hits if h.get("id") not in used_ids]
        for hit in candidates[:3]:
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
# 4. Guaranteed Silly Background Music Generator (Pure Python Synth)
# ---------------------------------------------------------------------------
def generate_silly_music_wav(filename: str, duration_sec: float, volume: float = 0.14):
    """
    Generates a quirky, bouncy pizzicato/8-bit comedy background track directly in Python.
    100% guaranteed to work without broken URLs or MoviePy version bugs!
    """
    print(f"Génération de la musique de fond comique ({duration_sec:.1f}s)...")
    sample_rate = 44100
    total_samples = int(sample_rate * duration_sec)

    # Bouncy staccato comedy melody notes (in Hz) + bassline
    melody_notes = [261.63, 329.63, 392.00, 523.25, 392.00, 329.63, 293.66, 349.23, 440.00, 392.00, 329.63, 293.66]
    bass_notes = [130.81, 196.00, 146.83, 196.00]
    step_sec = 0.22  # Fast, quirky comedic tempo
    step_samples = int(sample_rate * step_sec)

    with wave.open(filename, "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        chunk = bytearray()
        for i in range(total_samples):
            step_idx = i // step_samples
            pos_in_step = (i % step_samples) / sample_rate

            # Staccato plucky envelope (fades out quickly on each note like pizzicato strings)
            env_melody = math.exp(-pos_in_step * 14.0)
            freq_m = melody_notes[step_idx % len(melody_notes)]
            # Add a tiny comical pitch bend at the start of every note
            freq_m *= (1.0 + 0.03 * math.exp(-pos_in_step * 40.0))

            # Bouncy oom-pah bass every 2 steps
            freq_b = bass_notes[(step_idx // 2) % len(bass_notes)]
            env_bass = math.exp(-pos_in_step * 9.0) if (step_idx % 2 == 0) else 0.0

            # Rich harmonic wave for a goofy retro/cartoon vibe
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
# 5. Render Video (With Pacing Gaps & Guaranteed Audio Mixing)
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
    music_path = "background_music.wav"
    temp_files.append(music_path)

    try:
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

            voice_dur = max(0.5, raw_audio.duration - 0.05)
            audio_clip = mp(raw_audio, "subclipped", "subclip", 0, voice_dur)

            # Add a 0.35s pause at the end of each clip so he doesn't yap non-stop
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

        # Generate and mix the silly background music for the exact video duration
        generate_silly_music_wav(music_path, final_video.duration, volume=0.14)
        bg_music = AudioFileClip(music_path)
        to_close.append(bg_music)

        mixed_audio = CompositeAudioClip([final_video.audio, bg_music])
        final_video = mp(final_video, "with_audio", "set_audio", mixed_audio)

        print(f"Rendu final de la vidéo (Durée : {final_video.duration:.1f}s)...")
        final_video.write_videofile(output_path, fps=24, codec="libx264", audio_codec="aac")
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
    print("=== DÉMARRAGE DE L'AGENT (JOHN THE BLOB) ===")
    try:
        script_data = generate_script()
        print(f"Titre choisi : {script_data.title}")

        build_video(script_data)
        upload_to_youtube(script_data)
    except FatalError as e:
        print(f"ERREUR FATALE : {e}")
        sys.exit(1)
