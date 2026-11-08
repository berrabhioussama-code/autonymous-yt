import logging
import os
import random
import time
import urllib.parse
from pathlib import Path

import requests
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from moviepy.editor import (
    AudioFileClip,
    VideoFileClip,
    concatenate_videoclips,
)
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("autonymous_yt")

# Keep external calls bounded so a stalled API or network request can't hang the job forever.
HTTP_TIMEOUT = (10, 90)
PIXABAY_TIMEOUT = (10, 45)


class Scene(BaseModel):
    narration: str
    video_search_query: str


class VideoScript(BaseModel):
    title: str
    description: str
    tags: list[str]
    scenes: list[Scene] = Field(min_length=1)


def require_env(*names: str) -> None:
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"Missing required environment variable(s): {', '.join(missing)}")


def generate_script() -> VideoScript:
    require_env("GEMINI_API_KEY")
    logger.info("Initializing Gemini client")
    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"],
        http_options=types.HttpOptions(timeout=60_000),
    )
    niches = [
        "mind-blowing space facts",
        "bizarre historical events",
        "weird deep-sea creatures",
        "fascinating psychology tricks",
        "future technology breakthroughs",
    ]
    chosen_niche = random.choice(niches)
    logger.info("Generating a script for niche: %s", chosen_niche)

    prompt = f"""
    Create a viral 30-to-45-second YouTube Short script about {chosen_niche}.
    Provide a catchy title (under 70 characters, include #shorts), a brief SEO description, 5 tags, and 4 to 5 scenes.
    For each scene:
    - 'narration': 1-2 punchy sentences to be spoken aloud. You may include occasional Fish Audio emotion tags like [excited], [pause], or [emphasis] to make the narration expressive, but no emoji.
    - 'video_search_query': 1-2 broad English keywords to find a matching stock video on Pixabay (e.g., 'galaxy', 'ocean', 'brain', 'technology', 'nature').
    """

    # These are supported Gemini model IDs; the former 3.8 IDs do not resolve.
    models_to_try = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"]
    last_error = None
    for model_name in models_to_try:
        for attempt in range(1, 3):
            try:
                logger.info("Requesting script from %s (attempt %d/2)", model_name, attempt)
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=VideoScript,
                        temperature=0.9,
                    ),
                )
                if not response.text:
                    raise RuntimeError(f"Gemini returned an empty response for {model_name}")
                script = VideoScript.model_validate_json(response.text)
                logger.info("Script generated: %d scenes, title=%r", len(script.scenes), script.title)
                return script
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Gemini request failed (%s, attempt %d/2): %s: %s",
                    model_name,
                    attempt,
                    type(exc).__name__,
                    exc,
                )
                if attempt < 2:
                    time.sleep(attempt * 2)

    raise RuntimeError(f"All Gemini models failed. Last error: {last_error}") from last_error


def generate_voiceover(text: str, filename: str) -> None:
    require_env("FISH_API_KEY")
    url = "https://api.fish.audio/v1/tts"
    headers = {
        "Authorization": f"Bearer {os.environ['FISH_API_KEY']}",
        "Content-Type": "application/json",
        "model": "s2.1-pro-free",
    }
    payload = {
        "text": text,
        "format": "mp3",
        "mp3_bitrate": 128,
        "normalize": True,
    }
    voice_id = os.environ.get("FISH_VOICE_ID", "").strip()
    if voice_id:
        payload["reference_id"] = voice_id

    logger.info("Requesting voiceover (%d characters) -> %s", len(text), filename)
    response = requests.post(url, json=payload, headers=headers, timeout=HTTP_TIMEOUT)
    response.raise_for_status()
    if not response.content:
        raise RuntimeError("Fish Audio returned an empty audio file")
    Path(filename).write_bytes(response.content)
    logger.info("Voiceover saved (%d bytes): %s", len(response.content), filename)


def download_pixabay_video(query: str, filename: str) -> None:
    require_env("PIXABAY_API_KEY")
    api_key = os.environ["PIXABAY_API_KEY"]

    def search(search_query: str) -> list[dict]:
        params = {
            "key": api_key,
            "q": search_query,
            "per_page": 10,
            "safesearch": "true",
        }
        logger.info("Searching Pixabay for %r", search_query)
        response = requests.get(
            "https://pixabay.com/api/videos/",
            params=params,
            timeout=PIXABAY_TIMEOUT,
        )
        response.raise_for_status()
        return response.json().get("hits", [])

    hits = search(query)
    if not hits and query.strip().lower() != "space stars":
        logger.warning("No Pixabay results for %r; trying fallback search", query)
        hits = search("space stars")
    if not hits:
        raise RuntimeError(f"Pixabay returned no videos for {query!r} or the fallback query")

    videos = hits[min(random.randrange(min(3, len(hits))), len(hits) - 1)].get("videos", {})
    mp4_url = (
        videos.get("medium", {}).get("url")
        or videos.get("large", {}).get("url")
        or videos.get("small", {}).get("url")
    )
    if not mp4_url:
        raise RuntimeError(f"Pixabay result for {query!r} did not contain a video URL")

    logger.info("Downloading Pixabay video -> %s", filename)
    with requests.get(mp4_url, stream=True, timeout=PIXABAY_TIMEOUT) as response:
        response.raise_for_status()
        byte_count = 0
        with open(filename, "wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 256):
                if chunk:
                    output.write(chunk)
                    byte_count += len(chunk)
    if byte_count == 0:
        raise RuntimeError(f"Pixabay returned an empty video for {query!r}")
    logger.info("Video downloaded (%d bytes): %s", byte_count, filename)


def build_video(script: VideoScript, output_path="final_short.mp4") -> None:
    if not script.scenes:
        raise ValueError("Cannot build a video without scenes")

    clips = []
    audio_clips = []
    final_clip = None
    try:
        for i, scene in enumerate(script.scenes):
            audio_path = f"scene_{i}.mp3"
            video_path = f"scene_{i}.mp4"
            logger.info("Scene %d/%d: starting (query=%r)", i + 1, len(script.scenes), scene.video_search_query)

            generate_voiceover(scene.narration, audio_path)
            download_pixabay_video(scene.video_search_query, video_path)

            logger.info("Scene %d: opening media files", i + 1)
            audio_clip = AudioFileClip(audio_path)
            audio_clips.append(audio_clip)
            video_clip = VideoFileClip(video_path).without_audio()
            if video_clip.duration <= 0 or audio_clip.duration <= 0:
                raise RuntimeError(f"Scene {i + 1} has invalid audio/video duration")

            if video_clip.duration < audio_clip.duration:
                video_clip = video_clip.loop(duration=audio_clip.duration)
            else:
                video_clip = video_clip.subclip(0, audio_clip.duration)

            # Resize to cover a 9:16 canvas, then crop to a consistent output size.
            scale = max(1080 / video_clip.w, 1920 / video_clip.h)
            video_clip = video_clip.resize(scale)
            video_clip = video_clip.crop(
                x_center=video_clip.w / 2,
                y_center=video_clip.h / 2,
                width=1080,
                height=1920,
            )
            clips.append(video_clip.set_audio(audio_clip))
            logger.info(
                "Scene %d ready (audio %.1fs, video %.1fs)",
                i + 1,
                audio_clip.duration,
                video_clip.duration,
            )

        logger.info("Combining %d scenes", len(clips))
        final_clip = concatenate_videoclips(clips, method="compose")
        logger.info("Rendering %s (duration %.1fs)", output_path, final_clip.duration)
        final_clip.write_videofile(
            output_path,
            fps=24,
            codec="libx264",
            audio_codec="aac",
            logger="bar",
        )
        logger.info("Render complete: %s (%d bytes)", output_path, os.path.getsize(output_path))
    finally:
        if final_clip is not None:
            final_clip.close()
        for clip in clips:
            clip.close()
        for audio_clip in audio_clips:
            audio_clip.close()


def upload_to_youtube(script: VideoScript, video_path="final_short.mp4") -> None:
    require_env("YT_REFRESH_TOKEN", "YT_CLIENT_ID", "YT_CLIENT_SECRET")
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    logger.info("Authenticating with YouTube and preparing upload: %s", video_path)
    creds = Credentials(
        token=None,
        refresh_token=os.environ["YT_REFRESH_TOKEN"],
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.environ["YT_CLIENT_ID"],
        client_secret=os.environ["YT_CLIENT_SECRET"],
    )
    youtube = build("youtube", "v3", credentials=creds, cache_discovery=False)

    body = {
        "snippet": {
            "title": script.title[:100],
            "description": f"{script.description}\n\n#shorts",
            "tags": script.tags,
            "categoryId": "27",
        },
        "status": {
            "privacyStatus": "public",
            "selfDeclaredMadeForKids": False,
        },
    }

    media = MediaFileUpload(video_path, chunksize=8 * 1024 * 1024, resumable=True, mimetype="video/mp4")
    upload_request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    response = None
    while response is None:
        status, response = upload_request.next_chunk(num_retries=3)
        if status is not None:
            logger.info("YouTube upload progress: %.1f%%", status.progress() * 100)
    video_id = response.get("id")
    if not video_id:
        raise RuntimeError(f"YouTube upload response did not contain a video ID: {response}")
    logger.info("Upload successful! Video ID: %s", video_id)


if __name__ == "__main__":
    try:
        logger.info("Starting video generation pipeline")
        logger.info("Step 1/3: generating script")
        script_data = generate_script()
        logger.info("Generated title: %s", script_data.title)

        logger.info("Step 2/3: building video")
        build_video(script_data)

        logger.info("Step 3/3: uploading to YouTube")
        upload_to_youtube(script_data)
        logger.info("Pipeline completed successfully")
    except Exception:
        logger.exception("Pipeline failed")
        raise
