import re
import csv

PATTERN = r"([A-Za-zÀ-ÖØ-öø-ÿ0-9\.\'\s]+?)\s(?:vs|versus)\s([A-Za-zÀ-ÖØ-öø-ÿ0-9\.\'\s]+?)(?=\s*[-\|\]]|$)"

def aggiungi_indici(path_in, path_out):
    # Legge tutto il file
    with open(path_in, "r", encoding="utf-8") as f:
        contenuto = f.read()

    # Divide le battle (separate da righe vuote)
    battles = [blocco.strip() for blocco in re.split(r"\n\s*\n", contenuto) if blocco.strip()]

    nuove_battles = []
    for i, battle in enumerate(battles, start=1):
        righe = battle.splitlines()
        if not righe:
            continue
        # Prependi [Sn] al titolo (prima riga)
        titolo = righe[0]
        nuovo_titolo = f"[S{i}] {titolo}"
        righe[0] = nuovo_titolo
        nuove_battles.append("\n".join(righe))

    # Ricostruisce il file con righe vuote tra le battle
    nuovo_testo = "\n\n".join(nuove_battles)

    # Salva il risultato
    with open(path_out, "w", encoding="utf-8") as f:
        f.write(nuovo_testo)

    print(f"File aggiornato creato in: {path_out}")

def estrai_nomi_artisti(titolo):
    match = re.search(PATTERN, titolo, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return "UNK", "UNK"

def processa_dataset(path_file, path_csv):
    with open(path_file, "r", encoding="utf-8") as f:
        contenuto = f.read()

    battles = [blocco.strip() for blocco in re.split(r"\n\s*\n", contenuto) if blocco.strip()]

    with open(path_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        # intestazioni
        writer.writerow(["battle_id", "artist1", "artist2", "battle_name", "timestamp", "lyric"])

        for battle_id, battle in enumerate(battles, start=1):
            righe = battle.splitlines()
            if not righe:
                continue

            titolo = righe[0][6:-1]
            artista1, artista2 = estrai_nomi_artisti(titolo)
            battle_name = titolo.strip()

            for riga in righe[1:]:
                riga = riga.strip()
                if not riga:
                    continue

                match = re.match(r"\[(\d{2}:\d{2}(?::\d{2})?)\]\s*(.*)", riga)
                if match:
                    timestamp, testo = match.groups()
                else:
                    timestamp, testo = "", riga

                writer.writerow([battle_id, artista1, artista2, battle_name, timestamp, testo])

    print(f"CSV creato in: {path_csv}")

if __name__ == "__main__":
    indexed_dataset = "FreestyleAI/dataset_creation/prova_dataset_indexed.txt"
    aggiungi_indici(
        "FreestyleAI/dataset_creation/prova_dataset.txt",
        indexed_dataset
    )
    processa_dataset(indexed_dataset, "FreestyleAI/dataset_creation/output_battles.csv")