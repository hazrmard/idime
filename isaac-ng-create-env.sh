module purge && module load anaconda3/2021.05 && source $ANACONDA3_SH

loc=/lustre/isaac/scratch/$(whoami)
currloc=$(pwd)

echo "Location is $loc"

rm -rf $loc/env
rm $loc/env.tgz

conda create -p $loc/env --copy python=3.10
conda activate $loc/env

rm -rf $loc/deps
mkdir $loc/deps

pip download -r ./requirements.txt \
        -d $loc/deps \
        --index-url=https://download.pytorch.org/whl/cpu \
        --extra-index-url=https://pypi.org/simple

cd $loc
rm env.tgz deps.tgz
tar -cvzhf env.tgz env
tar -cvzhf deps.tgz deps
cd $currloc

# Also install locally
conda activate $loc/env
python -m pip install -r requirements.txt --find-links=$loc/deps --no-index
conda deactivate