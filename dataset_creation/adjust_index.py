import re
import sys
import uuid
from pathlib import Path


RX = re.compile(r"^battle_(\d+)\.[^.]+$", re.IGNORECASE)


def main():
    # Cartella di default: FreestyleAI/temporary_garbage/audio_eng rispetto a questo file
    repo_root = Path(__file__).resolve().parents[1]
    default_folder = repo_root / "temporary_garbage" / "audio_eng"
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else default_folder

    if not folder.is_dir():
        print(f"Errore: {folder} non è una cartella valida.")
        sys.exit(1)

    # Prendi tutti gli mp3
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".mp3"]
    if not files:
        print("Nessun file .mp3 trovato.")
        return

    # Ordina: prima quelli con indice numerico, per indice; poi gli altri per nome
    def sort_key(p: Path):
        m = RX.match(p.name)
        if m:
            return (0, int(m.group(1)))
        return (1, p.name.lower())

    files.sort(key=sort_key)

    # Crea mapping a battle_{i}.mp3 consecutivo da 1..N
    mappings = []
    for i, src in enumerate(files, start=1):
        dest = src.with_name(f"battle_{i}.mp3")
        if src.name != dest.name:
            mappings.append((src, dest))

    if not mappings:
        print("Tutti i file sono già in ordine e senza buchi.")
        return

    # Rinomina in due fasi per evitare collisioni
    tmp_tag = f".__tmp_{uuid.uuid4().hex}"
    temps = []
    for src, dest in mappings:
        tmp = src.with_name(src.name + tmp_tag)
        while tmp.exists():
            tmp = src.with_name(src.name + "." + uuid.uuid4().hex[:6] + tmp_tag)
        src.rename(tmp)
        temps.append((tmp, dest))

    for tmp, dest in temps:
        if dest.exists():
            dest.unlink()
        tmp.rename(dest)

    print(f"Rinominati {len(temps)} file in {folder}.")


if __name__ == "__main__":
    main()