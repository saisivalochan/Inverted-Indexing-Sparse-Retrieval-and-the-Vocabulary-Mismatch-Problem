# CS6101 assignment: inverted indexing, sparse retrieval, vocabulary mismatch

Run everything (one script at a time, in order) inside WSL:

    python download.py
    bash run_all.sh

`run_all.sh` runs `index.py`, `index_stats.py`, `phase2.py`, `phase3.py`, `phase4a.py`, `hyde_generate.py`,
`phase4b.py`, `phase5.py`, `make_tables.py`. Each script also accepts dataset names, e.g. `python phase2.py scifact`.
Results go to `results/`, LaTeX tables to `report/tables/`; compile `report/report.tex` (submission) and
`report/learning.tex` (study notes) with pdflatex.

| file | part |
|---|---|
| `datasets.py` | ir_datasets BEIR loaders |
| `common.py` | query loading, 500-query sample, ir_measures evaluation, result files |
| `index.py`, `index_stats.py` | Part 1: Lucene index, build time / size, IndexReader statistics |
| `phase2.py` | Part 2: default BM25, tuned BM25, TF-IDF |
| `phase3.py` | Part 3: BM25 failures, categories, Jaccard overlap |
| `rocchio.py`, `phase4a.py` | Part 4a: Rocchio and RM3 |
| `hyde_generate.py`, `phase4b.py` | Part 4b: HyDE |
| `phase5.py` | Part 5: SPLADE |
| `make_tables.py` | results -> LaTeX tables |
