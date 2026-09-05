import sys
import json
import time

from pyserini.search.lucene import LuceneSearcher

import common
from rocchio import Rocchio

for name in sys.argv[1:] or common.DATASETS:
    hyde = {d["qid"]: d["docs"] for d in map(json.loads, open(f"results/hyde_{name}.jsonl"))}
    queries, qrels = common.load(name, sample=common.SAMPLE)
    queries = [q for q in queries if q.query_id in hyde]
    qrels = [r for r in qrels if r.query_id in hyde]
    params = common.load_results("part2")[name]["tuned_params"]
    searcher = LuceneSearcher(f"indexes/{name}")
    searcher.set_bm25(params["k1"], params["b"])
    rocchio = Rocchio(f"indexes/{name}")
    res = {"n_queries": len(queries), "bm25": common.evaluate(common.search(searcher, queries), qrels)}

    start = time.time()
    hits = searcher.batch_search([q.text + " " + " ".join(hyde[q.query_id]) for q in queries], [q.query_id for q in queries], k=100, threads=8)
    res["concat"] = common.evaluate(common.hits_to_run(hits), qrels)
    res["concat_latency_ms"] = round(1000 * (time.time() - start) / len(queries), 1)

    for k in [10, 20]:
        start = time.time()
        run, expansions = {}, {}
        for q in queries:
            hits, expansions[q.query_id] = rocchio.search(searcher, q.text, rocchio.text_vectors(hyde[q.query_id]), k)
            run[q.query_id] = {h.docid: h.score for h in hits}
        res[f"rocchio_hyde_k={k}"] = common.evaluate(run, qrels)
        res[f"rocchio_hyde_k={k}_latency_ms"] = round(1000 * (time.time() - start) / len(queries), 1)
        res[f"example_terms_k={k}"] = {q.query_id: expansions[q.query_id] for q in queries[:12]}

    res["corpus_prf"] = {key: {"rocchio": v["rocchio"], "rm3": v["rm3"]} for key, v in common.load_results("part4a")[name]["settings"].items()}
    res["example_hyde"] = {q.query_id: {"query": q.text, "docs": hyde[q.query_id]} for q in queries[:3]}
    print(name, {k: v for k, v in res.items() if "example" not in k}, flush=True)
    common.save("part4b", name, res)
