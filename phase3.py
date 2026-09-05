import re
import sys
import statistics
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from pyserini.search.lucene import LuceneSearcher
from pyserini.index.lucene import LuceneIndexReader
from pyserini.analysis import Analyzer, get_lucene_analyzer

import common
import datasets

analyzer = Analyzer(get_lucene_analyzer())
ABBR = re.compile(r"\b[A-Z][A-Z0-9\-]{1,7}s?\b")


def has_expansion(abbr, text):
    letters = re.sub(r"[^A-Za-z]", "", abbr).lower()
    initials = "".join(w[0] for w in re.findall(r"[A-Za-z]+", text.lower()))
    return len(letters) >= 2 and letters in initials


def categorize(q_text, d_text, q_tok, d_tok):
    shared = q_tok & d_tok
    if not shared:
        return "no lexical overlap"
    q_abbr, d_abbr = set(ABBR.findall(q_text)), set(ABBR.findall(d_text))
    if any(has_expansion(a, d_text) for a in q_abbr - d_abbr) or any(has_expansion(a, q_text) for a in d_abbr - q_abbr):
        return "abbreviation vs expansion"
    if len(shared) / len(q_tok) < 0.5:
        return "partial overlap (paraphrase)"
    return "high overlap but outranked"


for name in sys.argv[1:] or common.DATASETS:
    queries, qrels = common.load(name)
    store = datasets.load_dataset(name).docs_store()
    params = common.load_results("part2")[name]["tuned_params"]
    searcher = LuceneSearcher(f"indexes/{name}")
    searcher.set_bm25(params["k1"], params["b"])
    reader = LuceneIndexReader(f"indexes/{name}")
    run = common.search(searcher, queries)
    gold = common.gold(qrels)

    pairs = []
    for q in queries:
        q_tok = set(analyzer.analyze(q.text))
        ranked = sorted(run[q.query_id], key=run[q.query_id].get, reverse=True)
        for docid in gold.get(q.query_id, {}):
            d_tok = set(reader.get_document_vector(docid) or {})
            rank = ranked.index(docid) + 1 if docid in ranked else None
            pairs.append({"query": q.text, "docid": docid, "rank": rank, "success": rank is not None and rank <= 10,
                          "jaccard": len(q_tok & d_tok) / len(q_tok | d_tok), "shared": sorted(q_tok & d_tok), "q_tok": q_tok, "d_tok": d_tok})

    succ = [p["jaccard"] for p in pairs if p["success"]]
    fail = [p["jaccard"] for p in pairs if not p["success"]]
    stats = {"pairs": len(pairs), "failure_rate": round(len(fail) / len(pairs), 4),
             "jaccard_success": {"mean": statistics.mean(succ), "median": statistics.median(succ)},
             "jaccard_failure": {"mean": statistics.mean(fail), "median": statistics.median(fail)},
             "auc": round(float(mannwhitneyu(succ, fail).statistic) / (len(succ) * len(fail)), 4), "failure_rate_by_bin": {}}
    for lo, hi in [(0, 0.0001), (0.0001, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 1.01)]:
        sel = [not p["success"] for p in pairs if lo <= p["jaccard"] < hi]
        if sel:
            stats["failure_rate_by_bin"][f"[{lo:.2f},{hi:.2f})"] = {"n": len(sel), "failure_rate": round(statistics.mean(sel), 4)}

    cats, examples = Counter(), defaultdict(list)
    for p in pairs:
        if not p["success"]:
            doc = store.get(p["docid"])
            cat = categorize(p["query"], doc.title + " " + doc.text, p["q_tok"], p["d_tok"])
            cats[cat] += 1
            if len(examples[cat]) < 3:
                examples[cat].append({"query": p["query"], "title": doc.title, "doc": doc.text[:300], "rank": p["rank"],
                                      "jaccard": round(p["jaccard"], 3), "shared": p["shared"]})
    stats["failure_categories"] = dict(cats)
    stats["examples"] = examples

    plt.figure(figsize=(5, 3.2))
    plt.hist([succ, fail], bins=20, range=(0, 0.6), density=True, label=[f"gold in top-10 (n={len(succ)})", f"gold missed (n={len(fail)})"])
    plt.xlabel("Jaccard(query tokens, gold-doc tokens)")
    plt.ylabel("density")
    plt.title(name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/part3_jaccard_{name}.pdf")
    print(name, {k: v for k, v in stats.items() if k != "examples"}, flush=True)
    common.save("part3", name, stats)
