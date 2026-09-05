import sys
import time

from pyserini.search.lucene import LuceneSearcher
from pyserini.pyclass import autoclass

import common

ClassicSimilarity = autoclass("org.apache.lucene.search.similarities.ClassicSimilarity")

for name in sys.argv[1:] or common.DATASETS:
    queries, qrels = common.load(name)
    searcher = LuceneSearcher(f"indexes/{name}")
    res = {"n_queries": len(queries)}

    searcher.set_bm25(0.9, 0.4)
    start = time.time()
    res["bm25_default"] = common.evaluate(common.search(searcher, queries), qrels)
    res["latency_ms"] = round(1000 * (time.time() - start) / len(queries), 2)

    grid = {}
    for k1 in [0.6, 0.9, 1.2, 1.5, 2.0]:
        for b in [0.3, 0.4, 0.5, 0.75, 1.0]:
            searcher.set_bm25(k1, b)
            grid[(k1, b)] = common.evaluate(common.search(searcher, queries), qrels)
            print(name, k1, b, grid[(k1, b)], flush=True)
    k1, b = max(grid, key=lambda p: grid[p]["nDCG@10"])
    res["bm25_tuned"] = grid[(k1, b)]
    res["tuned_params"] = {"k1": k1, "b": b}
    res["grid_nDCG@10"] = {f"{k1},{b}": m["nDCG@10"] for (k1, b), m in grid.items()}

    searcher.object.searcher.setSimilarity(ClassicSimilarity())
    res["tfidf"] = common.evaluate(common.search(searcher, queries), qrels)
    print(name, res, flush=True)
    common.save("part2", name, res)
