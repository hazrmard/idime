ENV_DIR=../env
DEPS=../deps
BIN_DIR=../bin

rm -rf $ENV_DIR
rm -rf $DEPS
rm -rf $BIN_DIR

module purge && module load anaconda3/2021.05 && source $ANACONDA3_SH

conda activate $ENV_DIR
python -m pip install -r requirements.txt --find-links=./$DEPS --no-index