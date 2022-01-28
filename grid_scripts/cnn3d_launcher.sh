#!/bin/bash
#int-or-string.sh
#SBATCH --job-name=CNN_3D
#SBATCH --account=def-joelzy
#SBATCH --time=30:00:00
#SBATCH --gres=gpu:1 --cpus-per-task=1
#SBATCH --mem-per-cpu=36000M
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
pip install --no-index scipy matplotlib tensorflow_gpu



python run_model_cnn3d.py $expDate $mdl_name $path_model_save_base $fname_data_train_val_test --path_existing_mdl=$path_existing_mdl --runOnCluster=$runOnCluster --chan1_n=$chan1_n --filt1_size=$filt1_size --filt1_3rdDim=$filt1_3rdDim --chan2_n=$chan2_n --filt2_size=$filt2_size --filt2_3rdDim=$filt2_3rdDim --chan3_n=$chan3_n --filt3_size=$filt3_size --filt3_3rdDim=$filt3_3rdDim --nb_epochs=$nb_epochs --thresh_rr=$thresh_rr --temporal_width=$temporal_width --pr_temporal_width=$pr_temporal_width --bz_ms=$bz_ms --BatchNorm=$BatchNorm --MaxPool=$MaxPool --c_trial=$c_trial --USE_CHUNKER=$use_chunker --trainingSamps_dur=$TRSAMPS
