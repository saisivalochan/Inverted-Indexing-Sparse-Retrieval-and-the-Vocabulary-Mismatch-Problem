from collections import Counter

from pyserini.search.lucene import querybuilder
from pyserini.search.lucene.querybuilder import JTerm, JTermQuery, JBooleanClauseOccur
from pyserini.index.lucene import LuceneIndexReader
from pyserini.analysis import Analyzer, get_lucene_analyzer


class Rocchio:
    def __init__(self, index_dir, alpha=1.0, beta=0.75, df_cutoff=0.10):
        self.reader = LuceneIndexReader(index_dir)
        self.n_docs = self.reader.stats()["documents"]
        self.analyzer = Analyzer(get_lucene_analyzer())
        self.alpha, self.beta, self.df_cutoff = alpha, beta, df_cutoff
        self.common = {}

    def too_common(self, term):
        if term not in self.common:
            self.common[term] = self.reader.get_term_counts(term, analyzer=None)[0] > self.df_cutoff * self.n_docs
        return self.common[term]

    def index_vectors(self, docids):
        return [self.reader.get_document_vector(d) or {} for d in docids]

    def text_vectors(self, texts):
        return [self.analyzer.compute_document_vector(t) for t in texts]

    def expand(self, query_text, vectors, k):
        q_tokens = self.analyzer.analyze(query_text)
        q, q_len, n = Counter(q_tokens), max(len(q_tokens), 1), max(len(vectors), 1)
        centroid = Counter()
        for vec in vectors:
            length = sum(vec.values()) or 1
            for t, tf in vec.items():
                centroid[t] += tf / length
        weights = {t: self.alpha * tf / q_len + self.beta / n * centroid.get(t, 0.0) for t, tf in q.items()}
        expansion = sorted((t for t in centroid if t not in q and not self.too_common(t)), key=lambda t: -centroid[t])[:k]
        for t in expansion:
            weights[t] = self.beta / n * centroid[t]
        return weights, expansion

    @staticmethod
    def boosted_query(weights):
        builder = querybuilder.get_boolean_query_builder()
        for term, w in weights.items():
            builder.add(querybuilder.get_boost_query(JTermQuery(JTerm("contents", term)), float(w)), JBooleanClauseOccur.should.value)
        return builder.build()

    def search(self, searcher, query_text, vectors, k, top=100):
        weights, expansion = self.expand(query_text, vectors, k)
        return searcher.search(self.boosted_query(weights), k=top), expansion
