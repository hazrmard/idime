from typing import Tuple
import pandas as pd
import numpy as np
import scipy
from sentence_transformers import SentenceTransformer, util

# sbert_data/

# model_name = 'all-MiniLM-L6-v2'
model_name = 'msmarco-distilbert-base-v4'



def get_save_path(dataset: str, cache_path: str='./bin'):
    return f'{cache_path}/sbert_{dataset}/'


def load_source(path: str):
    return pd.read_csv(path)


def train(df: pd.DataFrame, save_path: str='./bin/sbert'):
    model = SentenceTransformer(model_name, cache_folder=save_path)
    encodings = model.encode(df.Inspection.to_list(), convert_to_numpy=True)
    np.save(save_path + '/encodings.npy', encodings)
    return model,
    

def load_models(path: str='./bin/sbert'):
    model = SentenceTransformer(model_name, cache_folder=path)
    encodings = np.load(path + '/encodings.npy')
    return (model, encodings), 


def predict(query: str, model: Tuple[SentenceTransformer, np.ndarray], topn=5):
    model, encodings = model
    vector = model.encode(query)
    sims = util.semantic_search(vector[None,:], encodings, top_k=10)[0]
    indices = [res['corpus_id'] for res in sims]
    normed = scipy.special.softmax([res['score'] for res in sims])
    return sorted(
        [(idx, 0, n) for (idx, n) in zip(indices, normed)],
        key=lambda x: x[2], reverse=True
    )


if __name__=='__main__':
    import sys
    if len(sys.argv) > 1:
        data = sys.argv[1]
    else:
        data = 'maintnet'
    df = load_source('./data/' + data + '.csv')
    train(df, save_path=f'./bin/sbert_{data}')