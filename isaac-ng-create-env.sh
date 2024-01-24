module purge && module load anaconda3/2021.05 && source $ANACONDA3_SH

loc=/lustre/isaac/scratch/$(whoami)
currloc=$(pwd)

echo "Location is $loc"

rm -rf $loc/idime
rm $loc/idime.tgz

conda create -p $loc/idime --copy python=3.10
conda activate $loc/idime

rm -rf $loc/deps
mkdir $loc/deps

pip download -r ./requirements.txt \
        -d $loc/deps \
        --index-url=https://download.pytorch.org/whl/cpu \
        --extra-index-url=https://pypi.org/simple

cd $loc
rm idime.tgz deps.tgz
tar -cvzhf idime.tgz idime
tar -cvzhf deps.tgz deps
cd $currloc