import pickle
import numpy as np
import pandas as pd
import gensim
from sklearn.pipeline import Pipeline
from sklearn.manifold import Isomap
from sklearn.preprocessing import Normalizer
from sklearn.cluster import HDBSCAN
import bnlearn as bn


def load_source(path: str):
    return pd.read_csv(path)


def predict_insp_cluster(text, model):
    v = model['model_insp'].infer_vector(text)
    e = model['embedding'].transform([v])
    c = model['knn'].predict(e)
    return c[0]


def train(df: pd.DataFrame, save_path: str='./bin/clusters'):
    def read_documents(s: pd.Series, tokens_only=False):
        for i, line in enumerate(s):
            tokens = gensim.utils.simple_preprocess(str(line))
            if tokens_only:
                yield tokens
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

    documents_insp = list(read_documents(df.Inspection))
    documents_main = list(read_documents(df.Maintenance))

    # Using Doc2Vec, represent each document (each row in a column) as a vector
    # Two models are learned, each for inspection and maintenance documents
    VECTOR_SIZE = 32
    def make_model(corpus, vector_size=VECTOR_SIZE):
        model = gensim.models.doc2vec.Doc2Vec(vector_size=vector_size, min_count=2, epochs=40)
        model.build_vocab(corpus)
        model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
        return model

    model_insp = make_model(documents_insp)
    model_main = make_model(documents_main)

    # Use a manifold learning algorithm to reduce dimensionality for each document class
    DO_MANIFOLD = True
    if VECTOR_SIZE > 3 and DO_MANIFOLD:
        embedding = Isomap(n_neighbors=5, n_components=3, metric='cosine')
        x_insp = embedding.fit_transform(model_insp.dv.vectors)
        x_main = embedding.fit_transform(model_main.dv.vectors)
    else:
        x_insp = model_insp.dv.vectors.copy()
        x_main = model_main.dv.vectors.copy()
    
    # Cluster the embeddings
    pipe_insp = Pipeline([
        ('normalization', Normalizer()),
        ('clustering', HDBSCAN(min_cluster_size=100, metric='euclidean'))
        # ('clustering', DBSCAN(n_components=20))
    ])
    y_insp = pipe_insp.fit_predict(x_insp)

    pipe_main = Pipeline([
        ('normalization', Normalizer()),
        ('clustering', HDBSCAN(min_cluster_size=50, metric='euclidean'))
    ])
    y_main = pipe_main.fit_predict(x_main)

    # Discretize the data using cluster labels
    ddf = pd.DataFrame({'insp': y_insp, 'main': y_main}, index=df.index)

    # Learn a bayesian model, assuming a node structure
    edges = [list(ddf.columns),]
    dag = bn.make_DAG(edges)
    # parameter learning
    model = bn.parameter_learning.fit(dag, ddf)
    model = bn.independence_test(model, ddf, prune=False)

    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier()
    knn.fit(x_insp, y_insp)
    
    models = dict(
        embedding=embedding,
        model_insp=model_insp,
        # model_main=model_main,
        pipe_insp=pipe_insp,
        # pipe_main=pipe_main,
        knn=knn,
        model=model,
    )
    with open(save_path, 'wb') as f:
        pickle.dump(models, f)
    return models


def load_models(load_path: str='./bin/clusters'):
    with open(load_path, 'rb') as f:
        models = pickle.load(f)
    return models,


def predict(query: str, model: dict, topn=5):
    evidence = {
        'insp': predict_insp_cluster(gensim.utils.simple_preprocess(query), model),
        }
    res=bn.inference.fit(model['model'], variables=['main',],
                        evidence=evidence, verbose=0)
    resdf = res.df.set_index('main')
    
    examples = res.sample(topn).values.flatten()

    labels = model['pipe_insp'].named_steps['clustering'].labels_
    unique_labels, counts = np.unique(examples, return_counts=True)
    indices, clusters, scores = [], [], []
    for l, c in zip(unique_labels, counts):
        idx = np.arange(len(labels))[labels==l]
        idx = np.random.choice(idx, size=c)
        indices.extend(idx)
        clusters.extend([l]*c)
        scores.extend([resdf.p[l]/c] * c)
        # for i in idx:
        #     print(df.Maintenance.iloc[i])
    return sorted(zip(indices, clusters, scores), key=lambda x: x[2], reverse=True)


if __name__=='__main__':
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = 'maintnet'
    df = load_source('./data/' + path + '.csv')
    train(df, save_path='./bin/clusters_' + path)