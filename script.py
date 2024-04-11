import sys
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import torch
from torch.nn import CrossEntropyLoss
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=str, default='tmp')
parser.add_argument("--cache-size", type=int, default=512)
parser.add_argument("--n-start", type=int, default=4)
parser.add_argument("--enable-pos-shift", action='store_true')
args = parser.parse_args()

model_name = "meta-llama/Llama-2-7b-hf"
device = 'cuda'
loss_function = CrossEntropyLoss()

## make dir
output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)

## load model
tokenizer = AutoTokenizer.from_pretrained(model_name,
        trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
print(model.config)
model.eval()

## perplexity
max_length = model.config.max_position_embeddings
print(max_length)
with open("10146.txt", "r") as f: # first document of PG19 test set
    text = f.read()
encodings = tokenizer(text, return_tensors="pt")
corpus_seq_len = encodings.input_ids.size(1)

window = args.cache_size
f = open(output_dir / "logs.txt", "w")

# ---- window attention ----
if args.n_start < 1:
    past_key_values = None
    nlls = []
    for loc in range(0, corpus_seq_len-1):
        input_ids = encodings.input_ids[:, loc:loc+1].to(device)
        labels = encodings.input_ids[:, loc+1:loc+2]
        
        with torch.no_grad():
            # https://huggingface.co/docs/transformers/v4.39.3/en/model_doc/llama#transformers.LlamaModel
            outputs = model(input_ids,
                            past_key_values = past_key_values,
                            use_cache = True, # return past_key_values
                            ) # type(outputs): 'transformers.modeling_outputs.CausalLMOutputWithPast'
            
            logits = outputs.logits.reshape(-1, model.config.vocab_size)
            nll = loss_function(logits.detach().cpu(), labels.view(-1))
            print(f"loc: {loc}, nll: {nll.item():.2f}, ppl: {torch.exp(nll).item():.2f}", file=f, flush=True)

            past_seq_len = outputs.past_key_values[0][0].shape[2]
            if past_seq_len > window-1:
                past_key_values = list()
                # slice past_key_values
                for k, v in outputs.past_key_values: # each tuple has 2 tensors
                    k_slice = k[:, :, -window+1:, :]
                    v_slice = v[:, :, -window+1:, :]
                    past_key_values.append((k_slice, v_slice))
            else:
                past_key_values = outputs.past_key_values

        nlls.append(nll)
    f.close()
    ppl = torch.exp(torch.stack(nlls).mean())
    with open(output_dir / "ppl.txt", "w") as f:
        f.write(f"{ppl.item()}\n")

## ---- add initial tokens ----
else:
    if args.enable_pos_shift:
        from modify_llama import enable_llama_pos_shift_attention
        enable_llama_pos_shift_attention(model)
        
    past_key_values = None
    nlls = []
    for loc in range(0, corpus_seq_len-1):
        input_ids = encodings.input_ids[:, loc:loc+1].to(device)
        labels = encodings.input_ids[:, loc+1:loc+2]
        
        with torch.no_grad():
            # https://huggingface.co/docs/transformers/v4.39.3/en/model_doc/llama#transformers.LlamaModel
            outputs = model(input_ids,
                            past_key_values = past_key_values,
                            use_cache = True, # return past_key_values
                            ) # type(outputs): 'transformers.modeling_outputs.CausalLMOutputWithPast'
            
            logits = outputs.logits.reshape(-1, model.config.vocab_size)
            nll = loss_function(logits.detach().cpu(), labels.view(-1))
            print(f"loc: {loc}, nll: {nll.item():.2f}, ppl: {torch.exp(nll).item():.2f}", file=f, flush=True)

            past_seq_len = outputs.past_key_values[0][0].shape[2]
            if past_seq_len <= window-1:
                past_key_values = outputs.past_key_values
            else:
                past_key_values = list()
                # slice past_key_values
                for k, v in outputs.past_key_values: # each tuple has 2 tensors
                    k_slice = torch.cat((
                        k[:, :, :args.n_start, :],
                        k[:, :, -window+1+args.n_start:, :]
                    ), dim=2
                    )
                    v_slice = torch.cat((
                        v[:, :, :args.n_start, :],
                        v[:, :, -window+1+args.n_start:, :]
                    ), dim=2
                    )
                    past_key_values.append((k_slice, v_slice))
        nlls.append(nll)
    f.close()
    ppl = torch.exp(torch.stack(nlls).mean())
    with open(output_dir / "ppl.txt", "w") as f:
        f.write(f"{ppl.item()}\n")