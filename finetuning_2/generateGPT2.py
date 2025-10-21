from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

OUTPUT_DIR = "FreestyleAI/models/rap_finetuned"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carica modello e tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(OUTPUT_DIR)
model = GPT2LMHeadModel.from_pretrained(OUTPUT_DIR).to(device)

# Prompt di esempio
prompt = "<START> Yo, check it <LINE>"
inputs = tokenizer(
    prompt,
    return_tensors="pt",
    padding=True  # aggiunge padding se necessario
)
input_ids = inputs.input_ids.to(device)
attention_mask = inputs.attention_mask.to(device)

# Generazione
output = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,  # <- attenzione ai token di padding
    max_new_tokens=100,
    temperature=1.2,
    top_k=50,
    top_p=0.9,
    do_sample=True,
    pad_token_id=tokenizer.pad_token_id
)

print(tokenizer.decode(output[0], skip_special_tokens=False).replace("<LINE>", "\n").replace("<START>", "").strip())
