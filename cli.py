from argparse import ArgumentParser

import pandas as pd
import scipy
import gensim
from gensim.models.doc2vec import Doc2Vec
from gensim.utils import simple_preprocess

def load_source(path: str):
    return pd.read_csv(path)


def predict(query: str, model: Doc2Vec):
    vector = model.infer_vector(simple_preprocess(query))
    sims = model.dv.most_similar([vector], topn=10)
    normed = scipy.special.softmax([score for _,score in sims])
    return [(idx, n) for ((idx, _), n) in zip(sims, normed)]


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--mode', default='naive')
    args = parser.parse_args()

    if args.mode=='naive':
        model = gensim.models.Doc2Vec.load('./bin/doc2vec_insp')
        data = load_source('./data/maintnet.csv')

    while True:
        query = input('Enter query: ')
        answer = predict(query=query, model=model)
        for idx, score in answer:
            print('%.2f, %.2f, %s' % (score, data.iloc[idx].TimeCost, data.Maintenance.iloc[idx]))