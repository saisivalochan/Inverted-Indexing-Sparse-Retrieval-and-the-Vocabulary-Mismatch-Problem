import os
import json
import random

import datasets
from datasets import DATASETS
from ir_measures import nDCG, R, RR, AP, calc_aggregate, iter_calc

METRICS = [nDCG @ 10, R @ 100, RR @ 10, AP]
SAMPLE = 500


def load(name, sample=None):
    d = datasets.load_dataset(name)
    queries, qrels = list(datasets.get_queries(d)), list(datasets.get_qrels(d))
    if sample and len(queries) > sample:
        random.Random(0).shuffle(queries)
        queries = sorted(queries[:sample], key=lambda q: q.query_id)
        qids = {q.query_id for q in queries}
        qrels = [r for r in qrels if r.query_id in qids]
    return queries, qrels


def gold(qrels):
    g = {}
    for r in qrels:
        if r.relevance > 0:
            g.setdefault(r.query_id, {})[r.doc_id] = r.relevance
    return g


def evaluate(run, qrels):
    return {str(m): round(float(v), 4) for m, v in calc_aggregate(METRICS, qrels, run).items()}


def per_query(run, qrels):
    return {m.query_id: m.value for m in iter_calc([nDCG @ 10], qrels, run)}


def hits_to_run(hits):
    return {qid: {h.docid: h.score for h in hs} for qid, hs in hits.items()}


def search(searcher, queries, k=100):
    return hits_to_run(searcher.batch_search([q.text for q in queries], [q.query_id for q in queries], k=k, threads=8))


def load_results(part):
    path = f"results/{part}.json"
    return json.load(open(path)) if os.path.exists(path) else {}


def save(part, name, data):
    os.makedirs("results", exist_ok=True)
    results = load_results(part)
    results[name] = data
    json.dump(results, open(f"results/{part}.json", "w"), indent=2)
