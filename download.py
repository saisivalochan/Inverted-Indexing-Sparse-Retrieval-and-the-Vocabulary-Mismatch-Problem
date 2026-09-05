from huggingface_hub import snapshot_download
from pyserini.util import download_prebuilt_index

for model in ["naver/splade-cocondenser-ensembledistil", "Qwen/Qwen2.5-0.5B-Instruct"]:
    print("model", model, snapshot_download(model), flush=True)

for name in ["scifact", "fever", "hotpotqa"]:
    print("index", name, download_prebuilt_index(f"beir-v1.0.0-{name}.splade-pp-ed"), flush=True)
