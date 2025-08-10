import subprocess
import whisper, torch
import time, glob, os
import yt_dlp



start = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🚀  Device:", DEVICE)

# === PLAYLIST WITH BATTLES ===
'''playlist_urls = [
    "https://www.youtube.com/playlist?list=PLxcnTE5ZmNw70ahG9KhuxDMwpkGf6datY",
    "https://www.youtube.com/playlist?list=PLxcnTE5ZmNw4tVMDPGG9okG5zB_ciKO6N",
    "https://www.youtube.com/playlist?list=PLxcnTE5ZmNw7fpOWXehjrdE3oLdi2QKIS",
    "https://www.youtube.com/playlist?list=PLxcnTE5ZmNw6cD6nSQo6UZjg0X9aJcyAS",
    "https://www.youtube.com/playlist?list=PLxcnTE5ZmNw6fGQgyS4nBdauQbBkOvjqD",
    "https://www.youtube.com/playlist?list=PLxcnTE5ZmNw74ia_8uxYcfXNdA3KG01mq",
    "https://www.youtube.com/playlist?list=PLxcnTE5ZmNw5R_9X9rPH83OuiTUYpmx6w",
    "https://www.youtube.com/playlist?list=PLxcnTE5ZmNw5Ml0GgI8lEDJ05Syo1tvLr",
    "https://www.youtube.com/playlist?list=PLxcnTE5ZmNw5Me6KQeKvrLcLQOUHgveD3",
    "https://www.youtube.com/playlist?list=PLxcnTE5ZmNw6kR6M8AT5dtMcMPaqw5GuB",
    "https://www.youtube.com/playlist?list=PLxcnTE5ZmNw4Eg3Wvv7qhcd2QAxgFoYRd",
    "https://www.youtube.com/playlist?list=PLxcnTE5ZmNw6LGllc89XjZT_y1O2hFquG",
    "https://www.youtube.com/playlist?list=PLxcnTE5ZmNw57mCBllNTUro9BzU3cyV3w",
    "https://www.youtube.com/playlist?list=PLxcnTE5ZmNw6Um9DUTKyi8RSOmu6HEEo-",
    "https://www.youtube.com/playlist?list=PLxcnTE5ZmNw5NUCRwbzS9_of7hkpoUbMl",
    "https://www.youtube.com/playlist?list=PLxcnTE5ZmNw6KD2VM9X1LCFHYX7R2b6nw"
]

ydl_opts = {
    'quiet': True,
    'extract_flat': True,  # solo info senza scaricare
}



with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    for playlist_url in playlist_urls:
        info = ydl.extract_info(playlist_url, download=False)

        if 'entries' in info:
            for entry in info['entries']:
                if 'id' in entry:
                    video_urls = f"https://www.youtube.com/watch?v={entry['id']}"
                    youtube_urls.append(video_urls)
                else:
                    print("Entry senza id:", entry)
        else:
            print(f"Nessuna 'entries' trovata per {playlist_url}. Info keys:", info.keys())
            if 'id' in info:
                youtube_urls.append(f"https://www.youtube.com/watch?v={info['id']}")
'''

youtube_urls = []

with open("FreestyleAI/dataset_creation/youtubelinks.txt", "r") as f: 
    for line in f.readlines():
        youtube_urls.append(line[:-1])
        
#youtube_urls = youtube_urls[start:]
print(youtube_urls)
#loading medium model for faster transcription
model = whisper.load_model("medium", device = DEVICE)  


os.makedirs("FreestyleAI/temporary_garbage/audio_eng", exist_ok=True)


output_dataset = "FreestyleAI/dataset_creation/dataset_freestyle.txt"

#with open(output_dataset, "w", encoding="utf-8") as f:
#    pass


for i, url in enumerate(youtube_urls):
    start_time = time.time()
    print(f"\n=== [{i+1}/{len(youtube_urls)}] PROCESSING ===")

    audio_filename = f"FreestyleAI/temporary_garbage/audio_eng/battle_{start+i+1}.mp3"

    # download audio
    result = subprocess.run([
        "yt-dlp", "--cookies", "youtube_cookies.txt",
        "-x", "--audio-format", "mp3",
        "-o", audio_filename,
        url
    ])

    if result.returncode != 0:
        print(f"❌ Download fallito per {url}")
        continue

    if not os.path.exists(audio_filename):
        print(f"❌ Audio non trovato per il link {url}")
        continue

    # Prendi il titolo del video
    with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
        info = ydl.extract_info(url, download=False)
        video_title = info.get('title', 'NoTitle')

    print("-> Trascrizione in corso...")
    result = model.transcribe(audio_filename, word_timestamps=True, language="en")

    if "word_segments" in result:
        words = result["word_segments"]
        get_text = lambda w: w["text"].lower()
    else:
        words = []
        for segment in result.get("segments", []):
            if "words" in segment:
                words.extend(segment["words"])
        get_text = lambda w: w["word"].lower()

    PAUSE_THRESHOLD = 0.5
    lines = []
    current_line = []
    prev_end = None

    for w in words:
        if prev_end is not None and (w["start"] - prev_end) > PAUSE_THRESHOLD:
            line_start_time = current_line[0]["start"]
            line_text = " ".join([get_text(pw) for pw in current_line])
            line_text = " ".join(line_text.split())
            mins = int(line_start_time // 60)
            secs = int(line_start_time % 60)
            timestamp = f"[{mins:02d}:{secs:02d}]"
            lines.append(f"{timestamp} {line_text}")
            current_line = []
        current_line.append(w)
        prev_end = w["end"]

    if current_line:
        line_start_time = current_line[0]["start"]
        line_text = " ".join([get_text(pw) for pw in current_line])
        line_text = " ".join(line_text.split())
        mins = int(line_start_time // 60)
        secs = int(line_start_time % 60)
        timestamp = f"[{mins:02d}:{secs:02d}]"
        lines.append(f"{timestamp} {line_text}")

    with open(output_dataset, "a", encoding="utf-8") as f:
        f.write(f"<BATTLE> [{video_title}]\n")
        f.write("\n".join(lines))
        f.write("\n\n")

    print(f"✅ Trascrizione aggiunta a {output_dataset}")
    print(f"--- Tempo trascrizione --- {(time.time() - start_time) / 60:.2f} minuti")