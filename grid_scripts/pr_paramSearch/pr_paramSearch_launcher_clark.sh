#!/bin/bash
#int-or-string.sh
#SBATCH --job-name=prParamSearch_clark
#SBATCH --account=def-joelzy
#SBATCH --time=0:10:00
#SBATCH --gres=gpu:1 --cpus-per-task=1
#SBATCH --mem-per-cpu=16000M
#SBATCH -o logs/%j-out.txt
#SBATCH -e logs/%j-error.txt



module load python/3.6 cuda cudnn

cd $SLURM_TMPDIR
echo "directory changed"
echo $PWD
mkdir RetinaPredictors
cp -r ~/scratch/RetinaPredictors/grid_scripts/from_git/ $SLURM_TMPDIR/
cd $SLURM_TMPDIR/from_git
echo "directory changed"
echo $PWD
ls -l
echo "files copied"

#cp /home/sidrees/scratch/dynamic_retina/models/cluster_run_model_cnn3d.py $SLURM_TMPDIR/
#cp -R /home/sidrees/scratch/dynamic_retina/models/model $SLURM_TMPDIR/

virtualenv --no-download  $SLURM_TMPDIR/tensorflow_env
source $SLURM_TMPDIR/tensorflow_env
pip install --no-index scipy matplotlib tensorflow_gpu joblib



python pr_paramSearch.py $expDate $path_mdl $trainingDataset $testingDataset $path_excel $path_perFiles $lightLevel $pr_type $pr_mdl_name --samps_shift=$samps_shift --mdl_name=$mdl_name --c_beta=$c_beta --c_gamma=$c_gamma --c_tau_y=$c_tau_y --c_n_y=$c_n_y --c_tau_z=$c_tau_z --c_n_z=$c_n_z
