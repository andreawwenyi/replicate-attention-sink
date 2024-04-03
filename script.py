import sys
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import torch

# model_name = sys.argv[1]
model_name = "lmsys/vicuna-13b-v1.3"
device='cuda'

## load model
tokenizer = AutoTokenizer.from_pretrained(model_name,
        trust_remote_code=True,
)

with open("10146.txt", "r") as f:
    text = f.read()

encodings = tokenizer(text, return_tensors="pt")


model = AutoModelForCausalLM.from_pretrained(model_name, device=device)
print(model.config.model_type)
model.eval()

## perplexity
max_length = model.config.n_positions
window = 1024
seq_len = encodings.input_ids.size(1)

nlls = []
prev_end_loc = 0
for begin_loc in range(0, seq_len, window):
    end_loc = min(begin_loc + max_length, seq_len)
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)

        # loss is calculated using CrossEntropyLoss which averages over valid labels
        # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
        # to the left by 1.
        neg_log_likelihood = outputs.loss

    nlls.append(neg_log_likelihood)

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

ppl = torch.exp(torch.stack(nlls).mean())
