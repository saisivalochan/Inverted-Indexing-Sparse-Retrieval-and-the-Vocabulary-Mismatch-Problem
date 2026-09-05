import os
import sys
import json
import time
import subprocess
from collections import Counter

import torch
from pyserini.search.lucene import LuceneImpactSearcher
from pyserini.encode import SpladeQueryEncoder, SpladeDocumentEncoder
from pyserini.analysis import Analyzer, get_lucene_analyzer
from pyserini.util import download_prebuilt_index

import common

MODEL = "naver/splade-cocondenser-ensembledistil"
analyzer = Analyzer(get_lucene_analyzer())
encoder = SpladeQueryEncoder(MODEL, device="cpu")


def dir_size(path):
    return sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(path) for f in fs)


def encode_and_index(name):
    coll, index = f"collection_splade/{name}", f"indexes_splade/{name}"
    os.makedirs(coll, exist_ok=True)
    stats = common.load_results("part5").get(name, {}).get("own_index", {})
    if not os.path.exists(f"{coll}/docs.jsonl"):
        doc_encoder = SpladeDocumentEncoder(MODEL, device="cpu")
        docs = sorted((json.loads(l) for l in open(f"collection/{name}/{name}.jsonl")), key=lambda d: len(d["contents"]))
        start = time.time()
        with open(f"{coll}/docs.jsonl.tmp", "w") as f:
            for i in range(0, len(docs), 16):
                batch = docs[i:i + 16]
                with torch.no_grad():
                    vectors = doc_encoder.encode([d["contents"] for d in batch], max_length=256)
                for d, v in zip(batch, vectors):
                    f.write(json.dumps({"id": d["id"], "contents": "", "vector": {t: w for t, w in v.items() if w > 0}}) + "\n")
                print(f"encoded {i + len(batch)}/{len(docs)}", flush=True)
        os.rename(f"{coll}/docs.jsonl.tmp", f"{coll}/docs.jsonl")
        stats["encode_time_s"] = round(time.time() - start)
    start = time.time()
    subprocess.run(["python", "-m", "pyserini.index.lucene", "-collection", "JsonVectorCollection", "-input", coll,
                    "-index", index, "-impact", "-pretokenized"], check=True)
    stats.update(index_time_s=round(time.time() - start), index_size_bytes=dir_size(index))
    return index, stats


def splade_run(index, queries):
    searcher = LuceneImpactSearcher(index, encoder)
    start = time.time()
    hits = searcher.batch_search([q.text for q in queries], [q.query_id for q in queries], k=100, threads=8)
    return common.hits_to_run(hits), round(1000 * (time.time() - start) / len(queries), 1)


def splade_expansion(text):
    literal = set(encoder.tokenizer.tokenize(text))
    ranked = sorted(encoder.encode(text).items(), key=lambda x: -x[1])
    return [t for t, w in ranked if w > 0 and t not in literal and t.isalpha()][:10]


def stem(terms):
    return {analyzer.analyze(t)[0] for t in terms if analyzer.analyze(t)}


for name in sys.argv[1:] or common.DATASETS:
    queries, qrels = common.load(name, sample=common.SAMPLE)
    res = {"n_queries": len(queries)}
    if name == "scifact":
        index, res["own_index"] = encode_and_index(name)
        run, res["latency_ms"] = splade_run(index, queries)
        res["splade"] = common.evaluate(run, qrels)
        run, _ = splade_run(download_prebuilt_index(f"beir-v1.0.0-{name}.splade-pp-ed"), queries)
        res["splade_prebuilt_sanity"] = common.evaluate(run, qrels)
    else:
        index = download_prebuilt_index(f"beir-v1.0.0-{name}.splade-pp-ed")
        res["prebuilt_index_size_bytes"] = dir_size(index)
        run, res["latency_ms"] = splade_run(index, queries)
        res["splade"] = common.evaluate(run, qrels)
    print(name, "splade", res["splade"], flush=True)

    rocchio_terms = common.load_results("part4a")[name]["settings"]["N=10,k=20"]["example_terms"]
    hyde_terms = common.load_results("part4b")[name]["example_terms_k=20"]
    rows, overlap = [], Counter()
    for q in queries[:12]:
        s, r, h = splade_expansion(q.text), rocchio_terms[q.query_id][:10], hyde_terms[q.query_id][:10]
        S, R, H = stem(s), stem(r), stem(h)
        rows.append({"qid": q.query_id, "query": q.text, "splade": s, "rocchio": r, "hyde": h,
                     "splade&rocchio": sorted(S & R), "splade&hyde": sorted(S & H), "rocchio&hyde": sorted(R & H)})
        overlap.update({"splade&rocchio": len(S & R), "splade&hyde": len(S & H), "rocchio&hyde": len(R & H), "all3": len(S & R & H)})
    res["expansion_rows"] = rows
    res["expansion_overlap_avg"] = {k: round(v / len(rows), 2) for k, v in overlap.items()}
    print(name, "overlap", res["expansion_overlap_avg"], flush=True)
    common.save("part5", name, res)
