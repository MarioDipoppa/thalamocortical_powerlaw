#!/bin/bash
#$ -cwd
#$ -o joblog/TCE-batch.$JOB_ID
#$ -j y

# resources
#$ -l h_rt=0:03:00,h_data=2G
#$ -pe shared 1

# email (unchanged)
#$ -M $USER@mail
#$ -m n

### PACKED JOB ARRAY (100 per task)
#$ -t 1-1:1

### from Tim Lindsey

echo "Job $JOB_ID started on:   " `hostname -s`
echo "Job $JOB_ID started on:   " `date `
echo " "

# load environment
. /u/local/Modules/default/init/modules.sh
module load anaconda3
conda activate tf_torch_cpu

fname='../python_scripts/fcm_multinomial.py'
python $fname


# ---- PACKING LOOP ----
for i in `seq 0 999`; do
    my_task_id=$((SGE_TASK_ID + i))

    if [ $my_task_id -le 4950 ]; then
        echo "Running job index: $my_task_id"
        python $fname --task_id $my_task_id
    fi
done

echo "Job $JOB_ID ended on:   " `hostname -s`
echo "Job $JOB_ID ended on:   " `date `
echo " "