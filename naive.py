import pandas as pd
import scipy
import gensim
from gensim.models.doc2vec import Doc2Vec
from gensim.utils import simple_preprocess


def load_source(path: str):
    return pd.read_csv(path)


def load_models(path: str='./bin/doc2vec_insp'):
    model = gensim.models.Doc2Vec.load(path)
    return model,


def predict(query: str, model: Doc2Vec):
    vector = model.infer_vector(simple_preprocess(query))
    sims = model.dv.most_similar([vector], topn=10)
    normed = scipy.special.softmax([score for _,score in sims])
    return [(idx, n) for ((idx, _), n) in zip(sims, normed)]
