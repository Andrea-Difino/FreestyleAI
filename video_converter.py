import subprocess
import whisper
import glob
import os
import time

# === LISTA DEI VIDEO ===

start = 8

youtube_urls = [
    "https://youtu.be/oOsbHp0L1r0?si=jMMXeqM7Kp--Xh9C",
    "https://youtu.be/S1-1gqRnEOw?si=Jx_t7g__N3wzDRvz",
    "https://youtu.be/iWOiv0Q1yTM?si=izQs7NK-CkQhXGLK",
    "https://youtu.be/u6rRjp7SphA?si=Oo9PJySbz-Z71hKA",
    "https://youtu.be/6HHG2xPOYjs?si=5HnDOcQ4M0vsv3Oh",
    "https://youtu.be/_AUemdIQ3lk?si=0JURtC4TRvpELYmq",
    "https://youtu.be/dU8uVqqp_-g?si=ZxGn9Uffo6r6I07q"
]

python_venv = r".venv\\Scripts\\python.exe"

#loading medium model for faster transcription
model = whisper.load_model("medium", device="cuda")  


os.makedirs("audio", exist_ok=True)
os.makedirs("trascrizioni", exist_ok=True)


for i, url in enumerate(youtube_urls):
    start_time = time.time()
    print(f"\n=== [{i+1}/{len(youtube_urls)}] PROCESSING ===")

    audio_filename = f"audio/battle_{start+i+1}.mp3"

    # download audio from YouTube
    print("-> Scaricamento audio...")
    subprocess.run([
        python_venv, "-m", "yt_dlp", "-x", "--audio-format", "mp3",
        "-o", audio_filename, url
    ])

    # check if the audio file exists
    if not os.path.exists(audio_filename):
        print(f"❌ Audio non trovato per il link {url}")
        continue

    # transcription
    print("-> Trascrizione in corso...")
    result = model.transcribe(audio_filename, language="it")

    # save transcription to a text file
    txt_file = f"trascrizioni/trascrizione_{start+i+1}.txt"
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(result["text"])

    print(f"✅ Trascrizione salvata in {txt_file}")
    print(f"--- Tempo trascrizione --- {(time.time() - start_time):.2f / 60} minuti")
    print("GPU a riposo\n")
    time.sleep(210) # waiting 3.5 minutes to avoid GPU overload