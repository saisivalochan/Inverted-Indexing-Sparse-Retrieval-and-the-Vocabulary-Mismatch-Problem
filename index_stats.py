import sys

from pyserini.index.lucene import LuceneIndexReader

import common

for name in sys.argv[1:] or common.DATASETS:
    reader = LuceneIndexReader(f"indexes/{name}")
    stats = reader.stats()
    _, qrels = common.load(name)
    docid = qrels[0].doc_id
    vec = reader.get_document_vector(docid)
    top = sorted(vec.items(), key=lambda x: -x[1])[:5]
    stats["demo"] = {"docid": docid, "doc_length": sum(vec.values()), "distinct_terms": len(vec), "top_tf": top,
                     "df": {t: reader.get_term_counts(t, analyzer=None)[0] for t, _ in top}}
    common.save("part1_stats", name, stats)
    print(name, stats, flush=True)
