from argparse import ArgumentParser

import pandas as pd

def load_source(path: str):
    return pd.read_csv(path)


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--data', default='./data/maintnet.csv')
    parser.add_argument('--mode', default='naive')
    args = parser.parse_args()

    data = load_source(args.data)
    if args.mode=='naive':
        from naive import predict, load_models
        model, *_ = load_models()
    elif args.mode=='clusters':
        from naive import predict, load_models
        model, *_ = load_models()

    while True:
        query = input('Enter query: ')
        answer = predict(query=query, model=model)
        for idx, score in answer:
            print('%.2f, %.2f, %s' % (score, data.iloc[idx].TimeCost, data.Maintenance.iloc[idx]))