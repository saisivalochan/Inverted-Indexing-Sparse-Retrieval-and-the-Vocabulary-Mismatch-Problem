import ir_datasets

DATASETS = {
    "scifact": "beir/scifact/test",
    "fever": "beir/fever/test",
    "hotpotqa": "beir/hotpotqa/test",
}

def load_dataset(name):
    return ir_datasets.load(DATASETS[name])

def get_corpus(dataset):
    return dataset.docs_iter()

def get_queries(dataset):
    return dataset.queries_iter()

def get_qrels(dataset):
    return dataset.qrels_iter()