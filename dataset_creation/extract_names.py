import re

def estrai_nomi_artisti(titolo):
           
  pattern = r"([A-Za-zÀ-ÖØ-öø-ÿ0-9\.\'\s]+?)\s(?:vs|versus)\s([A-Za-zÀ-ÖØ-öø-ÿ0-9\.\'\s]+?)(?=\s*[-\|\]]|$)"
  match = re.search(pattern, titolo, flags=re.IGNORECASE)
  if match:
    return match.group(1).strip(), match.group(2).strip()
  return None


dataset = open("FreestyleAI/dataset_creation/dataset_freestyle.txt", "r").read()


canzoni = [blocco for blocco in dataset.strip().split("\n\n")]

# Ora canzoni è un array, ogni elemento è una canzone
titoli = [c.split("\n", 1)[0] for c in canzoni]
sfidanti = [estrai_nomi_artisti(titolo) for titolo in titoli]

for s in sfidanti:
    if s == None: 
       s = ("Unknown", "Unknown")
    print(s, "\n")
    #print(len(canzoni))

