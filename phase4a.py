import sys
import time

from pyserini.search.lucene import LuceneSearcher

import common
from rocchio import Rocchio

for name in sys.argv[1:] or common.DATASETS:
    queries, qrels = common.load(name, sample=common.SAMPLE)
    params = common.load_results("part2")[name]["tuned_params"]
    searcher = LuceneSearcher(f"indexes/{name}")
    searcher.set_bm25(params["k1"], params["b"])
    rocchio = Rocchio(f"indexes/{name}")

    base_run = common.search(searcher, queries)
    base_pq = common.per_query(base_run, qrels)
    res = {"n_queries": len(queries), "bm25": common.evaluate(base_run, qrels), "settings": {}}
    print(name, "bm25", res["bm25"], flush=True)

    for n_fb, k in [(5, 10), (10, 20)]:
        start = time.time()
        run, expansions = {}, {}
        for q in queries:
            fb_docs = [h.docid for h in searcher.search(q.text, k=n_fb)]
            hits, expansions[q.query_id] = rocchio.search(searcher, q.text, rocchio.index_vectors(fb_docs), k)
            run[q.query_id] = {h.docid: h.score for h in hits}
        latency = round(1000 * (time.time() - start) / len(queries), 1)
        pq = common.per_query(run, qrels)
        drops = sorted(queries, key=lambda q: pq.get(q.query_id, 0) - base_pq.get(q.query_id, 0))[:5]
        entry = {"rocchio": common.evaluate(run, qrels), "rocchio_latency_ms": latency,
                 "drift_candidates": [{"query": q.text, "bm25_ndcg10": round(base_pq.get(q.query_id, 0), 3),
                                       "rocchio_ndcg10": round(pq.get(q.query_id, 0), 3), "added": expansions[q.query_id]} for q in drops],
                 "example_terms": {q.query_id: expansions[q.query_id] for q in queries[:12]}}

        searcher.set_rm3(fb_terms=k, fb_docs=n_fb, original_query_weight=0.5)
        start = time.time()
        entry["rm3"] = common.evaluate(common.search(searcher, queries), qrels)
        entry["rm3_latency_ms"] = round(1000 * (time.time() - start) / len(queries), 1)
        searcher.unset_rm3()
        print(name, n_fb, k, "rocchio", entry["rocchio"], "rm3", entry["rm3"], flush=True)
        res["settings"][f"N={n_fb},k={k}"] = entry
    common.save("part4a", name, res)
