import pandas as pd
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset

# ------------------- Config -------------------
CSV_PATH = "FreestyleAI/dataset_creation/freestyle_dataset_kotd_clean.csv"
MODEL_NAME = "gpt2"
MAX_MODEL_LENGTH = 1024  # Dimensione massima blocco (GPT-2 supporta fino a 1024)
TRAIN_BATCH_SIZE = 2
NUM_EPOCHS = 5
LEARNING_RATE = 5e-5
OUTPUT_DIR = "FreestyleAI/models/rap_finetuned"
SPECIAL_TOKENS = ["<START>", "<END>", "<LINE>"]

# ------------------- Carica CSV -------------------
df = pd.read_csv(CSV_PATH)
print(f"Numero righe nel dataset: {len(df)}")
print(df.head())

# ------------------- Tokenizer -------------------
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
tokenizer.add_special_tokens({"pad_token": "<PAD>"})

# ------------------- Converti tutto in token IDs -------------------
all_token_ids = []

for _, group in df.groupby("battle_name"):
    battle_bars = group["bar"].tolist()
    
    # Inizia con START token
    battle_ids = [tokenizer.convert_tokens_to_ids("<START>")]
    
    # Aggiungi tutte le bars separate da LINE
    for bar in battle_bars:
        bar_ids = tokenizer.encode(bar, add_special_tokens=False)
        battle_ids.extend(bar_ids)
        battle_ids.append(tokenizer.convert_tokens_to_ids("<LINE>"))
    
    # Fine con END token
    battle_ids.append(tokenizer.convert_tokens_to_ids("<END>"))
    
    all_token_ids.extend(battle_ids)

print(f"Numero totale token IDs: {len(all_token_ids)}")

# ------------------- Dividi in blocchi di dimensione fissa -------------------
all_blocks = []
block_size = MAX_MODEL_LENGTH  # Usa la dimensione massima del modello

for i in range(0, len(all_token_ids), block_size):
    # Prendi esattamente block_size token (non +1)
    block = all_token_ids[i:i + block_size]
    # Assicurati che il blocco abbia almeno 2 token (per input e label)
    if len(block) >= 2:
        # PAD il blocco alla lunghezza massima se è più corto
        if len(block) < block_size:
            block = block + [tokenizer.pad_token_id] * (block_size - len(block))
        all_blocks.append(block)

print(f"Numero totale blocchi: {len(all_blocks)}")
if all_blocks:
    block_lengths = [len(block) for block in all_blocks]
    print(f"Lunghezza massima blocco: {max(block_lengths)}")
    print(f"Lunghezza minima blocco: {min(block_lengths)}")
    print(f"Lunghezza media blocco: {sum(block_lengths) / len(block_lengths):.2f}")

# ------------------- Crea dataset HF -------------------
input_ids_list = [b[:-1] for b in all_blocks]
labels_list = [b[1:] for b in all_blocks]

# Aggiungi padding anche a input_ids e labels per mantenere la lunghezza uniforme
input_ids_list = [ids + [tokenizer.pad_token_id] * (MAX_MODEL_LENGTH - 1 - len(ids)) 
                  if len(ids) < MAX_MODEL_LENGTH - 1 else ids[:MAX_MODEL_LENGTH - 1] 
                  for ids in input_ids_list]

labels_list = [ids + [-100] * (MAX_MODEL_LENGTH - 1 - len(ids))  # -100 per ignorare il padding nella loss
               if len(ids) < MAX_MODEL_LENGTH - 1 else ids[:MAX_MODEL_LENGTH - 1] 
               for ids in labels_list]

# Verifica che tutte le sequenze abbiano la stessa lunghezza
print(f"\n=== Verifica lunghezze ===")
input_lengths = set(len(ids) for ids in input_ids_list)
label_lengths = set(len(ids) for ids in labels_list)
print(f"Lunghezze uniche input_ids: {input_lengths}")
print(f"Lunghezze uniche labels: {label_lengths}")

split_idx = int(0.8 * len(input_ids_list))
train_dataset = Dataset.from_dict({
    "input_ids": input_ids_list[:split_idx],
    "labels": labels_list[:split_idx]
})
val_dataset = Dataset.from_dict({
    "input_ids": input_ids_list[split_idx:],
    "labels": labels_list[split_idx:]
})
print(f"Train blocks: {len(train_dataset)}, Val blocks: {len(val_dataset)}")

# ------------------- Modello -------------------
# Determina il device (GPU se disponibile, altrimenti CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando device: {device}")

model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model.resize_token_embeddings(len(tokenizer))

# Configura la lunghezza massima del modello PRIMA di qualsiasi operazione
model.config.max_position_embeddings = MAX_MODEL_LENGTH
# Assicurati che il tokenizer usi la stessa lunghezza massima
tokenizer.model_max_length = MAX_MODEL_LENGTH

# Muovi il modello sul device corretto
model.to(device)

# ------------------- Data collator SENZA padding (già fatto manualmente) -------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False
)

# ------------------- Fine-tuning -------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    report_to="none",
    save_total_limit=3,
    no_cuda=False,  # Usa GPU se disponibile
    use_cpu=False,  # Non forzare CPU
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

trainer.train()

# Salva il modello finale
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nModello salvato in: {OUTPUT_DIR}")

# ------------------- Test di generazione -------------------
prompt = "<START> Yo, check it <LINE>"
# Limita esplicitamente la lunghezza del prompt
inputs = tokenizer(
    prompt, 
    padding='max_length', 
    max_length=MAX_MODEL_LENGTH,
    truncation=True,
    return_tensors="pt"
)
input_ids = inputs.input_ids.to(device)
attention_mask = inputs.attention_mask.to(device)

output = model.generate(
    input_ids,
    attention_mask=attention_mask,  # <- importante
    max_new_tokens=50,
    temperature=1.2,  # aumenta creatività
    top_k=50,
    top_p=0.9,
    do_sample=True,
    pad_token_id=tokenizer.pad_token_id  # usa <PAD> invece di EOS
)

print("\n=== Test di generazione ===")
print(tokenizer.decode(output[0], skip_special_tokens=False))