import os
import sys
import json
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import common

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SETTINGS = {"scifact": (2, 96), "fever": (1, 64), "hotpotqa": (1, 64)}
BATCH = 8
PROMPT = {
    "scifact": "Write a short passage from a scientific paper abstract that provides evidence for or against this claim.\nClaim: {q}\nPassage:",
    "fever": "Write a short Wikipedia-style passage that verifies or refutes this claim.\nClaim: {q}\nPassage:",
    "hotpotqa": "Write a short Wikipedia-style passage that answers this question.\nQuestion: {q}\nPassage:",
}

torch.set_num_threads(4)
tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()

for name in sys.argv[1:] or common.DATASETS:
    queries, _ = common.load(name, sample=common.SAMPLE)
    n_docs, max_new_tokens = SETTINGS[name]
    out = f"results/hyde_{name}.jsonl"
    done = {json.loads(l)["qid"] for l in open(out)} if os.path.exists(out) else set()
    todo = [q for q in queries if q.query_id not in done]
    with open(out, "a") as f:
        for i in range(0, len(todo), BATCH):
            batch = todo[i:i + BATCH]
            start = time.time()
            prompts = [tok.apply_chat_template([{"role": "user", "content": PROMPT[name].format(q=q.text)}], tokenize=False, add_generation_prompt=True) for q in batch]
            inputs = tok(prompts, return_tensors="pt", padding=True)
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9,
                                     num_return_sequences=n_docs, pad_token_id=tok.pad_token_id)
            texts = tok.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            for j, q in enumerate(batch):
                f.write(json.dumps({"qid": q.query_id, "query": q.text, "docs": [s.strip() for s in texts[j * n_docs:(j + 1) * n_docs]]}) + "\n")
            f.flush()
            print(f"{name}: {i + len(batch)}/{len(todo)} queries, {time.time() - start:.0f}s per batch", flush=True)
