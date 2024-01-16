# iDIME

## Installation

Installation requires `conda` or the required python version.

```bash
env_name=idime
conda create -n $env_name python=3.10
conda activate $env_name
bash install-deps.sh
```

## Running

First, the models need to be trained on data. Running one of the modules will do that. The training process expects a csv file with the following columns at least:

```
Inspection  Maintenance TimeCost    ... any other cols
string      string      float
```

Then, train the models by calling the modules. THey will create files under `./bin/`

```bash
python clusters.py maintnet # maintnet is the name of the csv file under ./data
python naive.py maintnet
```

The command line interface is provided via `cli.py`. It can use logic defined in other modules with the trained models.

```bash
python cli.py --mode clusters

# interactive loop starts
```