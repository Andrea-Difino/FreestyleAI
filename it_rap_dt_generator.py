import re

import lyricsgenius
import pandas as pd
import time

from tqdm import tqdm

# Inserisci qui il tuo token personale da https://genius.com/developers
GENIUS_ACCESS_TOKEN = "qpLxtkAW7D1QPFANE1jeJr3XzGvDND1NeGpBWmgI3gtF3pZhI0aQi_xdtA3S4LEh"

# Inizializza Genius
genius = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN, skip_non_songs=True, excluded_terms=["(Remix)", "(Live)"])
genius.verbose = False  # Disabilita output eccessivo

# Lista di artisti rap italiani
italian_rappers = [
    "Gemitaiz", "Nitro", "Ernia", "Rkomi", "MadMan", "J-Ax",
    "Sfera Ebbasta", "Ghali", "Noyz Narcos", "Tedua", "Izi"
]
#"Salmo", "Marracash", "Guè", "Fabri Fibra", "Lazza", "Inoki",
# Parametri
MAX_SONGS_PER_ARTIST = 30
data = []


def clean_lyrics(text):
    # Trova la prima parentesi quadra e taglia tutto prima
    match = re.search(r'\[.*?\]', text)
    if match:
        text = text[match.start():]  # taglia tutto prima della prima [

    # Rimuove tutto ciò che è tra parentesi quadre (es: [Intro], [Ritornello: artista])
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)

    # Rimuove righe vuote o solo spazi
    text = '\n'.join([line.strip() for line in text.split('\n') if line.strip()])

    return text

# Loop sugli artisti
for artist_name in italian_rappers:
    print(f"\n▶️ Scarico canzoni di: {artist_name}")
    try:
        artist = genius.search_artist(artist_name, max_songs=MAX_SONGS_PER_ARTIST, sort="popularity")
        if artist:
            for song in artist.songs:
                print(song.title)
                raw_lyrics = clean_lyrics(song.lyrics)
                # Suddividi in righe vere (barra per barra)
                lines = [line.strip() for line in raw_lyrics.split('\n') if line.strip()]

                for line in lines:
                    data.append({
                        "artist": artist.name,
                        "title": song.title,
                        "lyric": line  # attenzione: ora è UNA SOLA riga (barra)
                    })
        time.sleep(2)  # Rispetta il rate limit
    except Exception as e:
        print(f"❌ Errore con {artist_name}: {e}")
        continue

# Salva in CSV
df = pd.DataFrame(data)
df.to_csv("italian_rap_lyrics.csv", index=False)
print(f"\n✅ Dataset salvato: italian_rap_lyrics.csv — Totale brani: {len(df)}")