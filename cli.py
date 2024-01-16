from argparse import ArgumentParser

import pandas as pd

def load_source(path: str):
    return pd.read_csv(path)


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--data', default='maintnet')
    parser.add_argument('--mode', default='naive')
    args = parser.parse_args()
    print('Using data source %s and approach %s' % (args.data, args.mode))
    data = load_source('./data/' + args.data + '.csv')
    if args.mode=='naive':
        from naive import predict, load_models
        model, *_ = load_models('./bin/naive_' + args.data)
    elif args.mode=='clusters':
        from clusters import predict, load_models
        model, *_ = load_models('./bin/clusters_' + args.data)

    while True:
        query = input('\nRecommend Actions for: ')
        answer = predict(query=query, model=model, topn=5)
        print('Recommended Actions:')
        print('--------------------')
        for i, (idx, cluster, score) in enumerate(answer):
            print('%2d.\t%s\n\tTime Cost: %.2f hours' % (i+1, data.Maintenance.iloc[idx], data.iloc[idx].TimeCost))