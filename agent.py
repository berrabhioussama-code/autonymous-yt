import os
import random
import urllib.parse
import requests
from pydantic import BaseModel
from google import genai
from google.genai import types
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

class Scene(BaseModel):
    narration: str
    video_search_query: str

class VideoScript(BaseModel):
    title: str
    description: str
    tags: list[str]
    scenes: list[Scene]

def generate_script() -> VideoScript:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    niches = [
        "mind-blowing space facts",
        "bizarre historical events",
        "weird deep-sea creatures",
        "fascinating psychology tricks",
        "future technology breakthroughs"
    ]
    chosen_niche = random.choice(niches)
    
    prompt = f"""
    Create a viral 30-to-45-second YouTube Short script about {chosen_niche}.
    Provide a catchy title (under 70 characters, include #shorts), a brief SEO description, 5 tags, and 4 to 5 scenes.
    For each scene:
    - 'narration': 1-2 punchy sentences to be spoken aloud. You may include occasional Fish Audio emotion tags like [excited], [pause], or [emphasis] to make the narration expressive, but no emojis or stage directions.
    - 'video_search_query': 1-2 broad English keywords to find a matching stock video on Pixabay (e.g., 'galaxy', 'ocean', 'brain', 'technology', 'nature').
    """
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=VideoScript,
            temperature=0.9,
        ),
    )
    return VideoScript.model_validate_json(response.text)

def generate_voiceover(text: str, filename: str):
    url = "https://api.fish.audio/v1/tts"
    headers = {
        "Authorization": f"Bearer {os.environ['FISH_API_KEY']}",
        "Content-Type": "application/json",
        "model": "s2.1-pro-free"
    }
    payload = {
        "text": text,
        "format": "mp3",
        "mp3_bitrate": 128,
        "normalize": True
    }
    
    voice_id = os.environ.get("FISH_VOICE_ID", "").strip()
    if voice_id:
        payload["reference_id"] = voice_id
        
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    
    with open(filename, "wb") as f:
        f.write(response.content)

def download_pixabay_video(query: str, filename: str):
    api_key = os.environ["PIXABAY_API_KEY"]
    encoded_query = urllib.parse.quote(query)
    url = f"https://pixabay.com/api/videos/?key={api_key}&q={encoded_query}&per_page=10&safesearch=true"
    r = requests.get(url).json()
    
    hits = r.get("hits", [])
    if not hits:
        fallback_url = f"https://pixabay.com/api/videos/?key={api_key}&q=space+stars&per_page=10&safesearch=true"
        hits = requests.get(fallback_url).json().get("hits", [])
        
    chosen_video = random.choice(hits[:3])["videos"]
    mp4_url = (
        chosen_video.get("medium", {}).get("url")
        or chosen_video.get("large", {}).get("url")
        or chosen_video["small"]["url"]
    )
    
    with open(filename, "wb") as f:
        f.write(requests.get(mp4_url).content)

def build_video(script: VideoScript, output_path="final_short.mp4"):
    clips = []
    for i, scene in enumerate(script.scenes):
        audio_path = f"scene_{i}.mp3"
        video_path = f"scene_{i}.mp4"
        
        print(f"Generating scene {i+1}: {scene.video_search_query}")
        generate_voiceover(scene.narration, audio_path)
        download_pixabay_video(scene.video_search_query, video_path)
        
        audio_clip = AudioFileClip(audio_path)
        video_clip = VideoFileClip(video_path).without_audio()
        
        if video_clip.duration < audio_clip.duration:
            video_clip = video_clip.loop(duration=audio_clip.duration)
        else:
            video_clip = video_clip.subclip(0, audio_clip.duration)
            
        video_clip = video_clip.resize(height=1920)
        if video_clip.w > 1080:
            video_clip = video_clip.crop(x_center=video_clip.w / 2, width=1080, height=1920)
            
        video_clip = video_clip.set_audio(audio_clip)
        clips.append(video_clip)
        
    final_clip = concatenate_videoclips(clips, method="compose")
    final_clip.write_videofile(output_path, fps=24, codec="libx264", audio_codec="aac")

def upload_to_youtube(script: VideoScript, video_path="final_short.mp4"):
    creds = Credentials(
        token=None,
        refresh_token=os.environ["YT_REFRESH_TOKEN"],
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.environ["YT_CLIENT_ID"],
        client_secret=os.environ["YT_CLIENT_SECRET"]
    )
    youtube = build("youtube", "v3", credentials=creds)
    
    body = {
        "snippet": {
            "title": script.title[:100],
            "description": f"{script.description}\n\n#shorts",
            "tags": script.tags,
            "categoryId": "27"
        },
        "status": {
            "privacyStatus": "public",
            "selfDeclaredMadeForKids": False
        }
    }
    
    media = MediaFileUpload(video_path, chunksize=-1, resumable=True, mimetype="video/mp4")
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    response = request.execute()
    print(f"Uploaded successfully! Video ID: {response['id']}")

if __name__ == "__main__":
    print("1. Writing script with Gemini...")
    script_data = generate_script()
    print(f"Title: {script_data.title}")
    
    print("2. Building video with Pixabay clips and Fish Audio voiceover...")
    build_video(script_data)
    
    print("3. Uploading to YouTube...")
    upload_to_youtube(script_data)
