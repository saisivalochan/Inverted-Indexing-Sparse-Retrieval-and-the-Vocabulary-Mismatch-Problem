import os
import sys
import json
import time
import subprocess

import datasets
from datasets import DATASETS
import common

for name in sys.argv[1:] or DATASETS:
    collection_dir, index_dir = f"collection/{name}", f"indexes/{name}"
    os.makedirs(collection_dir, exist_ok=True)
    os.makedirs(index_dir, exist_ok=True)
    jsonl = f"{collection_dir}/{name}.jsonl"
    if not os.path.exists(jsonl):
        with open(jsonl, "w") as f:
            for doc in datasets.get_corpus(datasets.load_dataset(name)):
                f.write(json.dumps({"id": doc.doc_id, "contents": doc.title + "\n" + doc.text}) + "\n")

    start = time.time()
    subprocess.run(["python", "-m", "pyserini.index.lucene", "-collection", "JsonCollection",
                    "-input", collection_dir, "-index", index_dir, "-storeDocvectors"], check=True)
    build_time = time.time() - start
    size = sum(os.path.getsize(f"{index_dir}/{f}") for f in os.listdir(index_dir))
    common.save("part1", name, {"build_time_s": round(build_time), "index_size_bytes": size})
    print(name, round(build_time), "s", size // 1000000, "MB")
