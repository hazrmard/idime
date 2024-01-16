import pandas as pd
import scipy
import gensim
from gensim.models.doc2vec import Doc2Vec
from gensim.utils import simple_preprocess


def load_source(path: str):
    return pd.read_csv(path)


def train(df: pd.DataFrame, save_path: str='./bin/naive'):
    # Converting the dataframe in to 2 corpora of documents for gensim.
    # gensim will independently process each class of documents (inspection, maintenance)
    # for later analysis.
    def read_documents(s: pd.Series, tokens_only=False):
            for i, line in enumerate(s):
                tokens = gensim.utils.simple_preprocess(line.lower())
                if tokens_only:
                    yield tokens
                else:
                    # For training data, add tags
                    yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

    documents_insp = list(read_documents(df.Inspection))

    # Using Doc2Vec, represent each document (each row in a column) as a vector
    # Two models are learned, each for inspection and maintenance documents
    VECTOR_SIZE = 32
    def make_model(corpus, vector_size=VECTOR_SIZE):
        model = gensim.models.doc2vec.Doc2Vec(vector_size=vector_size, min_count=2, epochs=40)
        model.build_vocab(corpus)
        model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
        return model

    model_insp = make_model(documents_insp)
    model_insp.save(save_path)
    return model_insp,


def load_models(path: str='./bin/naive'):
    model = gensim.models.Doc2Vec.load(path)
    return model,


def predict(query: str, model: Doc2Vec):
    vector = model.infer_vector(simple_preprocess(query))
    sims = model.dv.most_similar([vector], topn=10)
    normed = scipy.special.softmax([score for _,score in sims])
    return [(idx, 0, n) for ((idx, _), n) in zip(sims, normed)]


if __name__=='__main__':
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = 'maintnet'
    df = load_source('./data/' + path + '.csv')
    train(df, save_path='./bin/naive_' + path)