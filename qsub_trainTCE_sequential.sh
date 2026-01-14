#$ -N train-TCE
#$ -cwd
#$ -V
#$ -l h_rt=12:00:00,h_data=16G,
#$ -j y
#$ -o joblog/train_TCE.$JOB_ID
#$ -M $USER@mail

# -V flag inherits the current environment variables, so just activate the environment

/u/home/s/skirti/miniforge3/envs/tce_env/bin/python train.py \
    --params /u/home/s/skirti/project-mdipoppa/thalamocortical-expansion/02_code/thalamocortical_powerlaw/params.yml \
    --data /u/home/s/skirti/project-mdipoppa/thalamocortical-expansion/01_data/digits/triplet_digits_2.mat \
    --out /u/home/s/skirti/project-mdipoppa/thalamocortical-expansion/01_data/models \
    --seed 1234

echo "sleep"
sleep 1m