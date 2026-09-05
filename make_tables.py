import os
import json
import shutil

import common

OUT = "report/tables"
os.makedirs(OUT, exist_ok=True)
DS = ["scifact", "fever", "hotpotqa"]
DSNAME = {"scifact": "SciFact", "fever": "FEVER", "hotpotqa": "HotpotQA"}
M = ["nDCG@10", "R@100", "RR@10", "AP"]
MHEAD = "nDCG@10 & R@100 & MRR@10 & MAP"


def esc(s):
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
                 ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    return s


def fmt(m):
    return " & ".join(f"{m[k]:.4f}" if k in m else "--" for k in M)


def write(fname, text):
    with open(f"{OUT}/{fname}", "w", encoding="utf-8") as f:
        f.write(text)


def table(header, rows, caption, label, colspec):
    body = "\n".join(r + r" \\" for r in rows)
    return (f"\\begin{{table}}[ht]\\centering\\small\n\\begin{{tabular}}{{{colspec}}}\\toprule\n{header} \\\\ \\midrule\n"
            f"{body}\n\\bottomrule\\end{{tabular}}\n\\caption{{{caption}}}\\label{{{label}}}\\end{{table}}\n")


def num(x):
    return f"{x:,}" if isinstance(x, int) else str(x)


def sizes(b):
    return f"{b / 1e6:.1f} MB" if b < 1e9 else f"{b / 1e9:.2f} GB"


p1, p1s, p2 = common.load_results("part1"), common.load_results("part1_stats"), common.load_results("part2")
rows = []
for d in DS:
    s, st = p1.get(d, {}), p1s.get(d, {})
    rows.append(f"{DSNAME[d]} & {num(st.get('documents', '--'))} & {p2.get(d, {}).get('n_queries', '--')} & {num(st.get('unique_terms', '--'))} & "
                f"{s.get('build_time_s', 0) / 60:.1f} min & {sizes(s.get('index_size_bytes', 0))}")
write("part1.tex", table("Dataset & Documents & Test queries & Unique terms & Build time & Index size", rows,
                         "Part 1: Lucene index per dataset (Pyserini \\texttt{JsonCollection}, \\texttt{-storeDocvectors}, 4 indexing threads).", "tab:part1", "lrrrrr"))

rows = []
for d in DS:
    dm = p1s.get(d, {}).get("demo")
    if dm:
        top = ", ".join(f"{esc(t)} (tf={tf}, df={dm['df'][t]:,})" for t, tf in dm["top_tf"])
        rows.append(f"{DSNAME[d]} & {esc(dm['docid'])} & {dm['doc_length']} & {dm['distinct_terms']} & {top}")
write("part1_demo.tex", table("Dataset & Gold doc id & Length & Distinct terms & Top-tf terms (tf, corpus df)", rows,
                              "Part 1: IndexReader statistics for the first gold document of each dataset (analysed, stemmed terms).", "tab:p1demo", "llrrp{0.45\\textwidth}"))

rows = []
for d in DS:
    r = p2.get(d)
    if not r:
        continue
    tp = r["tuned_params"]
    rows.append(f"\\multirow{{3}}{{*}}{{{DSNAME[d]}}} & BM25 default ($k_1{{=}}0.9, b{{=}}0.4$) & {fmt(r['bm25_default'])}")
    rows.append(f" & BM25 tuned ($k_1{{=}}{tp['k1']}, b{{=}}{tp['b']}$) & {fmt(r['bm25_tuned'])}")
    rows.append(f" & TF-IDF (Lucene \\texttt{{ClassicSimilarity}}) & {fmt(r['tfidf'])}")
    rows.append("\\midrule")
rows = [r for r in rows][:-1] if rows else rows
write("part2.tex", table(f"Dataset & Model & {MHEAD}", rows, "Part 2: sparse baselines on the full BEIR test query sets.", "tab:part2", "llrrrr"))
rows = []
for d in DS:
    r = p2.get(d)
    if not r:
        continue
    g = r["grid_nDCG@10"]
    best, worst = max(g, key=g.get), min(g, key=g.get)
    rows.append(f"{DSNAME[d]} & {g[best]:.4f} ({best}) & {g[worst]:.4f} ({worst}) & {g['0.9,0.4']:.4f} & {r['latency_ms']}")
write("part2_grid.tex", table("Dataset & Best nDCG@10 ($k_1$,$b$) & Worst nDCG@10 ($k_1$,$b$) & Default & BM25 latency (ms/query)", rows,
                              "Part 2: grid search over $k_1\\in\\{0.6,0.9,1.2,1.5,2.0\\}$, $b\\in\\{0.3,0.4,0.5,0.75,1.0\\}$ on all test queries (nDCG@10); latency = batch BM25 search with 8 threads.", "tab:grid", "lrrrr"))

p3 = common.load_results("part3")
rows, cat_rows, bin_rows, ex = [], [], [], []
CATS = ["no lexical overlap", "abbreviation vs expansion", "partial overlap (paraphrase)", "high overlap but outranked"]
for d in DS:
    r = p3.get(d)
    if not r:
        continue
    rows.append(f"{DSNAME[d]} & {r['pairs']} & {r['failure_rate']:.3f} & {r['jaccard_success']['mean']:.3f} / {r['jaccard_success']['median']:.3f} & "
                f"{r['jaccard_failure']['mean']:.3f} / {r['jaccard_failure']['median']:.3f} & {r['auc']:.3f}")
    c = r["failure_categories"]
    tot = sum(c.values())
    cat_rows.append(f"{DSNAME[d]} & " + " & ".join(f"{c.get(k, 0)} ({100 * c.get(k, 0) / max(tot, 1):.0f}\\%)" for k in CATS) + f" & {tot}")
    b = r["failure_rate_by_bin"]
    bin_rows.append(f"{DSNAME[d]} & " + " & ".join(f"{b[k]['failure_rate']:.2f} ({b[k]['n']})" if k in b else "--" for k in
                                                  ["[0.00,0.00)", "[0.00,0.05)", "[0.05,0.10)", "[0.10,0.20)", "[0.20,1.01)"]))
    ex.append(f"\\paragraph{{{DSNAME[d]}}}\n\\begin{{itemize}}\\setlength\\itemsep{{0pt}}")
    for cat in CATS:
        for e in r["examples"].get(cat, [])[:2]:
            rank = f"rank {e['rank']}" if e["rank"] else "not in top-100"
            ex.append(f"\\item[] \\textbf{{{esc(cat)}}} ({rank}, J={e['jaccard']}): \\emph{{Q:}} {esc(e['query'])} "
                      f"\\emph{{Gold:}} {esc(e['title'])} --- {esc(e['doc'][:200])}\\ldots\\ (shared: {esc(', '.join(e['shared']) or 'none')})")
    ex.append("\\end{itemize}")
write("part3_stats.tex", table("Dataset & (q,gold) pairs & Failure rate & Jaccard success (mean/med) & Jaccard failure (mean/med) & AUC", rows,
                               "Part 3: BM25 (tuned) failure = gold document not in the top-10.  Jaccard overlap of analysed query and gold-document tokens; "
                               "AUC = probability that a random success pair has higher overlap than a random failure pair.", "tab:p3stats", "lrrrrr"))
write("part3_cats.tex", table("Dataset & no overlap & abbrev.\\ vs exp.\\ & partial overlap & high overlap, outranked & total", cat_rows,
                              "Part 3: failure cases by (heuristic) category.", "tab:p3cats", "lrrrrr"))
write("part3_bins.tex", table("Dataset & $J{=}0$ & $0<J<.05$ & $.05\\le J<.10$ & $.10\\le J<.20$ & $J\\ge.20$", bin_rows,
                              "Part 3: failure rate (number of pairs) per Jaccard-overlap bin.", "tab:p3bins", "lrrrrr"))
write("part3_examples.tex", "\n".join(ex))
for d in DS:
    if os.path.exists(f"results/part3_jaccard_{d}.pdf"):
        shutil.copy(f"results/part3_jaccard_{d}.pdf", OUT)
write("part3_fig.tex", "\\begin{figure}[ht]\\centering\n" + "\n".join(
    f"\\includegraphics[width=0.32\\textwidth]{{tables/part3_jaccard_{d}.pdf}}" for d in DS if d in p3) +
    "\n\\caption{Part 3: distribution of Jaccard overlap for pairs where BM25 finds the gold document in the top-10 vs.\\ misses it.}\\label{fig:p3}\\end{figure}\n")

p4 = common.load_results("part4a")
rows, drift = [], []
for d in DS:
    r = p4.get(d)
    if not r:
        continue
    n = 1 + 2 * len(r["settings"])
    rows.append(f"\\multirow{{{n}}}{{*}}{{{DSNAME[d]} ({r['n_queries']} q)}} & BM25 tuned & {fmt(r['bm25'])}")
    for key, e in r["settings"].items():
        rows.append(f" & Rocchio {key} & {fmt(e['rocchio'])}")
    for key, e in r["settings"].items():
        rows.append(f" & RM3 {key} & {fmt(e['rm3'])}")
    rows.append("\\midrule")
    drift.append(f"\\paragraph{{{DSNAME[d]}}}\\begin{{itemize}}\\setlength\\itemsep{{0pt}}")
    for key, e in r["settings"].items():
        for c in e["drift_candidates"][:2]:
            drift.append(f"\\item[] [{key}] \\emph{{{esc(c['query'])}}} --- nDCG@10 {c['bm25_ndcg10']} $\\to$ {c['rocchio_ndcg10']}; added: {esc(', '.join(c['added']))}")
    drift.append("\\end{itemize}")
write("part4a.tex", table(f"Dataset & Model & {MHEAD}", rows[:-1],
                          "Part 4a: own Rocchio ($\\alpha{=}1,\\beta{=}0.75$, df-cutoff 10\\%) and Pyserini RM3 (original-query weight 0.5) at two $(N,k)$ settings, "
                          "vs.\\ tuned BM25 on the same queries (FEVER / HotpotQA: fixed random sample of 500 test queries).", "tab:p4a", "llrrrr"))
write("part4a_drift.tex", "\n".join(drift))

p4b = common.load_results("part4b")
rows, hy = [], []
for d in DS:
    r = p4b.get(d)
    if not r:
        continue
    prf = r["corpus_prf"]
    best_r = max(prf, key=lambda k: prf[k]["rocchio"]["nDCG@10"]) if prf else None
    best_m = max(prf, key=lambda k: prf[k]["rm3"]["nDCG@10"]) if prf else None
    rows.append(f"\\multirow{{6}}{{*}}{{{DSNAME[d]} ({r['n_queries']} q)}} & BM25 tuned (no feedback) & {fmt(r['bm25'])}")
    rows.append(f" & HyDE naive concatenation & {fmt(r['concat'])}")
    rows.append(f" & HyDE + Rocchio ($k{{=}}10$) & {fmt(r['rocchio_hyde_k=10'])}")
    rows.append(f" & HyDE + Rocchio ($k{{=}}20$) & {fmt(r['rocchio_hyde_k=20'])}")
    if best_r:
        rows.append(f" & corpus PRF: Rocchio {best_r} & {fmt(prf[best_r]['rocchio'])}")
        rows.append(f" & corpus PRF: RM3 {best_m} & {fmt(prf[best_m]['rm3'])}")
    rows.append("\\midrule")
    for qid, e in list(r["example_hyde"].items())[:1]:
        hy.append(f"\\paragraph{{{DSNAME[d]}}} \\emph{{Q:}} {esc(e['query'])} \\emph{{Hypothetical document:}} {esc(e['docs'][0][:400])}\\ldots")
write("part4b.tex", table(f"Dataset & Variant & {MHEAD}", rows[:-1],
                          "Part 4b: LLM-generated feedback (Qwen2.5-0.5B-Instruct, 2 hypothetical documents per query) vs.\\ corpus PRF (best 4a setting), same queries as Part 4a.",
                          "tab:p4b", "llrrrr"))
write("part4b_examples.tex", "\n".join(hy))

p5 = common.load_results("part5")
rows, exp_rows, ov_rows = [], [], []
for d in DS:
    r = p5.get(d)
    if not r:
        continue
    base = p4.get(d, {}).get("bm25") if r["n_queries"] == p4.get(d, {}).get("n_queries") else p2.get(d, {}).get("bm25_tuned")
    src = "own index" if "own_index" in r else "prebuilt splade-pp-ed"
    rows.append(f"\\multirow{{2}}{{*}}{{{DSNAME[d]} ({r['n_queries']} q)}} & BM25 tuned & {fmt(base) if base else '--'}")
    rows.append(f" & SPLADE++ ED ({src}) & {fmt(r['splade'])}")
    if "splade_prebuilt_sanity" in r:
        rows.append(f" & SPLADE++ ED (prebuilt, sanity check) & {fmt(r['splade_prebuilt_sanity'])}")
    rows.append("\\midrule")
    o = r["expansion_overlap_avg"]
    ov_rows.append(f"{DSNAME[d]} & {len(r['expansion_rows'])} & {o['splade&rocchio']} & {o['splade&hyde']} & {o['rocchio&hyde']} & {o['all3']}")
    for e in r["expansion_rows"][:10]:
        exp_rows.append(f"\\multicolumn{{4}}{{l}}{{\\textbf{{{DSNAME[d]} {esc(e['qid'])}}}: {esc(e['query'][:110])}}} \\\\")
        exp_rows.append(f" & {esc(', '.join(e['splade']))} & {esc(', '.join(e['rocchio']))} & {esc(', '.join(e['hyde']))} \\\\ \\addlinespace[2pt]")
write("part5.tex", table(f"Dataset & Model & {MHEAD}", rows[:-1],
                         "Part 5: SPLADE++ EnsembleDistil (\\texttt{naver/splade-cocondenser-ensembledistil}) vs.\\ tuned BM25 on the same queries.", "tab:p5", "llrrrr"))
write("part5_overlap.tex", table("Dataset & queries & SPLADE$\\cap$Rocchio & SPLADE$\\cap$HyDE & Rocchio$\\cap$HyDE & all three", ov_rows,
                                 "Part 5: average number of shared expansion terms (top-10 per source, compared after Porter stemming).", "tab:p5ov", "lrrrrr"))
write("part5_terms.tex", "\\begin{small}\\begin{longtable}{p{0.02\\textwidth}p{0.3\\textwidth}p{0.3\\textwidth}p{0.3\\textwidth}}\\toprule\n"
      " & SPLADE (beyond query word-pieces) & Rocchio (4a, corpus PRF) & HyDE + Rocchio (4b) \\\\ \\midrule\\endhead\n" + "\n".join(exp_rows) +
      "\n\\bottomrule\\caption{Part 5: top-10 expansion terms per source for the first 10 sample queries per dataset (Rocchio/HyDE terms are Porter-stemmed).}\\label{tab:p5terms}\\end{longtable}\\end{small}\n")

rows = []
for d in DS:
    r = p5.get(d)
    if not r:
        continue
    if "own_index" in r:
        o = r["own_index"]
        rows.append(f"{DSNAME[d]} & own (CPU encoding) & {o.get('encode_time_s', 0) / 60:.1f} min & {o['index_time_s']:.0f} s & {sizes(o['index_size_bytes'])} & {r['latency_ms']}")
    else:
        rows.append(f"{DSNAME[d]} & prebuilt & -- & -- & {sizes(r['prebuilt_index_size_bytes'])} & {r['latency_ms']}")
write("part5_index.tex", table("Dataset & Impact index & Encoding time & Index time & Index size & Query latency (ms)", rows,
                               "Part 5: SPLADE impact indexes (latency = end-to-end per query incl.\\ CPU query encoding, 8 search threads).", "tab:p5idx", "llrrrr"))
print("tables written to", OUT)
