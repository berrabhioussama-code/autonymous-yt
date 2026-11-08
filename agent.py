import os
import time
import random
import urllib.parse
import requests
from pydantic import BaseModel
from google import genai
from google.genai import types

print("DEBUG: Chargement des bibliothèques...")
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
print("DEBUG: Bibliothèques chargées avec succès.")

class Scene(BaseModel):
    narration: str
    video_search_query: str

class VideoScript(BaseModel):
    title: str
    description: str
    tags: list[str]
    scenes: list[Scene]

def generate_script() -> VideoScript:
    print("1. Connexion à Gemini...")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
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
    
    models_to_try = ["gemini-3.8-flash", "gemini-3.8-flash-lite"]
    for model_name in models_to_try:
        try:
            print(f"Tentative avec le modèle {model_name}...")
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=VideoScript,
                    temperature=0.9,
                ),
            )
            print("Script généré avec succès par Gemini !")
            return VideoScript.model_validate_json(response.text)
        except Exception as e:
            print(f"Échec avec {model_name}: {e}")
            time.sleep(2)
            
    raise RuntimeError("Tous les modèles Gemini ont échoué.")

def generate_voiceover(text: str, filename: str):
    print(f"Génération de la voix off pour : {text[:30]}...")
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
    print(f"Téléchargement de la vidéo Pixabay pour la requête : '{query}'...")
    api_key = os.environ["PIXABAY_API_KEY"]
    encoded_query = urllib.parse.quote(query)
    url = f"https://pixabay.com/api/videos/?key={api_key}&q={encoded_query}&per_page=10&safesearch=true"
    r = requests.get(url).json()
    
    hits = r.get("hits", [])
    if not hits:
        fallback_url = f"https://pixabay.com/api/videos/?key={api_key}&q=gym+motivation&per_page=10&safesearch=true"
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
        
        print(f"--- Traitement de la scène {i+1}/{len(script.scenes)} ---")
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
        
    print("Assemblage final de la vidéo avec MoviePy...")
    final_clip = concatenate_videoclips(clips, method="compose")
    final_clip.write_videofile(output_path, fps=24, codec="libx264", audio_codec="aac")
    print("Vidéo prête !")

def upload_to_youtube(script: VideoScript, video_path="final_short.mp4"):
    print("Envoi de la vidéo sur YouTube...")
    creds = Credentials(
        token=None,
        refresh_token=os.environ["YT_REFRESH_TOKEN"].strip(),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.environ["YT_CLIENT_ID"].strip(),
        client_secret=os.environ["YT_CLIENT_SECRET"].strip()
    )
    youtube = build("youtube", "v3", credentials=creds)
    
    body = {
        "snippet": {
            "title": script.title[:100],
            "description": f"{script.description}\n\n#shorts #motivation",
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
    print(f"Envoyé avec succès ! ID de la vidéo : {response['id']}")

if __name__ == "__main__":
    print("=== DÉMARRAGE DE L'AGENT YOUTUBE ===")
    script_data = generate_script()
    print(f"Titre choisi : {script_data.title}")
    
    build_video(script_data)
    upload_to_youtube(script_data)
