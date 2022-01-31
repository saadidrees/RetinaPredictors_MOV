#!/bin/bash
#int-or-string.sh


# updating repo
cd from_git
git fetch --all
git pull
cd ..
echo "Fetched latest files from git"

PARAMS_FILE="model_params.csv"
LOG_DIR="/home/sidrees/scratch/RetinaPredictors_MOV/grid_scripts/logs"

if [ ! -d "$LOG_DIR" ]; then 
	mkdir $LOG_DIR
fi


expDate_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f1) )
mdl_name_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f2) )
path_model_save_base_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f3) )
fname_data_train_val_test_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f4) )
path_existing_mdl_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f5) )
thresh_rr_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f6) )
temporal_width_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f7) )
pr_temporal_width_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f8) )
bz_ms_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f9) )
nb_epochs_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f10) )
chan1_n_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f11) )
filt1_size_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f12) )
filt1_3rdDim_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f13) )
chan2_n_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f14) )
filt2_size_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f15) )
filt2_3rdDim_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f16) )
chan3_n_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f17) )
filt3_size_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f18) )
filt3_3rdDim_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f19) )
BatchNorm_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f20) )
MaxPool_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f21) )
num_trials_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f22) )
use_chunker_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f23 )
TRSAMPS_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f24) )
lr_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f25) )


numParams=${#expDate_all[@]}
echo "Number of parameter combinations: $numParams"

typeset -i runOnCluster=1

#for ((i=0; i<$numParams; i++));
for ((i=0; i<1; i++));
do
 expDate=${expDate_all[i]}
 mdl_name=${mdl_name_all[i]}
 path_model_save_base=${path_model_save_base_all[i]}
# path_model_save_base=$path_model_save_base/$expDate
 fname_data_train_val_test=${fname_data_train_val_test_all[i]}
 path_existing_mdl=${path_existing_mdl_all[i]}
 
 thresh_rr=${thresh_rr_all[i]}
 typeset -i temporal_width=${temporal_width_all[i]}
 typeset -i pr_temporal_width=${pr_temporal_width_all[i]}
 typeset -i bz_ms=${bz_ms_all[i]}
 typeset -i nb_epochs=${nb_epochs_all[i]}
 
 typeset -i chan1_n=${chan1_n_all[i]}
 typeset -i filt1_size=${filt1_size_all[i]}
 typeset -i filt1_3rdDim=${filt1_3rdDim_all[i]}
 
 typeset -i chan2_n=${chan2_n_all[i]}
 typeset -i filt2_size=${filt2_size_all[i]}
 typeset -i filt2_3rdDim=${filt2_3rdDim_all[i]}
 
 typeset -i chan3_n=${chan3_n_all[i]}
 typeset -i filt3_size=${filt3_size_all[i]}
 typeset -i filt3_3rdDim=${filt3_3rdDim_all[i]}
 
 typeset -i BatchNorm=${BatchNorm_all[i]}
 typeset -i MaxPool=${MaxPool_all[i]}
 
 typeset -i num_trials=${num_trials_all[i]}
 typeset -i use_chunker=${use_chunker_all[i]}
 typeset -i TRSAMPS=${TRSAMPS_all[i]}
 lr=${lr_all[i]}

 

 echo "expDate: $expDate"
 echo "mdl_name: $mdl_name"
 echo "path_model_save_base: $path_model_save_base"
 echo "fname_data_train_val_test: $fname_data_train_val_test"
 echo "path_existing_mdl_all: $path_existing_mdl_all"
 echo "thresh_rr: $thresh_rr"
 echo "temporal_width: $temporal_width"
 echo "pr_temporal_width: $pr_temporal_width" 
 echo "bz_ms: $bz_ms"
 echo "nb_epochs: $nb_epochs"
 echo "chan1_n: $chan1_n"
 echo "filt1_size: $filt1_size"
 echo "filt1_3rdDim: $filt1_3rdDim"
 echo "chan2_n: $chan2_n"
 echo "filt2_size: $filt2_size"
 echo "filt2_3rdDim: $filt2_3rdDim"
 echo "chan3_n: $chan3_n"
 echo "filt3_size: $filt3_size"
 echo "filt3_3rdDim: $filt3_3rdDim" 
 echo "BatchNorm: $BatchNorm"
 echo "MaxPool: $MaxPool" 
 echo "num_trials: $num_trials" 
 echo "USE_CHUNKERS: $use_chunker" 
 echo "TRSAMPS: $TRSAMPS" 
 echo "Learning rate: $lr"
 
 for ((t=1; t<$num_trials+1; t++));
 do
  typeset -i c_trial=$t
  JOB_ID=$(sbatch --export=LOG_DIR=$LOG_DIR,expDate=$expDate,mdl_name=$mdl_name,path_model_save_base=$path_model_save_base,fname_data_train_val_test=$fname_data_train_val_test,path_existing_mdl=$path_existing_mdl,runOnCluster=$runOnCluster,chan1_n=$chan1_n,filt1_size=$filt1_size,filt1_3rdDim=$filt1_3rdDim,chan2_n=$chan2_n,filt2_size=$filt2_size,filt2_3rdDim=$filt2_3rdDim,chan3_n=$chan3_n,filt3_size=$filt3_size,filt3_3rdDim=$filt3_3rdDim,nb_epochs=$nb_epochs,thresh_rr=$thresh_rr,temporal_width=$temporal_width,pr_temporal_width=$pr_temporal_width,bz_ms=$bz_ms,BatchNorm=$BatchNorm,MaxPool=$MaxPool,c_trial=$c_trial,use_chunker=$use_chunker,TRSAMPS=$TRSAMPS,lr=$lr cnn3d_launcher.sh)
 
  echo $JOB_ID
  JOB_ID=$(echo "$JOB_ID" | grep -Eo '[0-9]{1,8}')
 
#  echo "JOB ID: $A\n\nexpDate: $expDate\nthresh_rr: $thresh_rr\ntemporal_width: $temporal_width\nbz_ms: $bz_ms\nnb_epochs: $nb_epochs\nchan1_n: $chan1_n\nfilt1_size: $filt1_size\nfilt1_3rdDim: $filt1_3rdDim\nfilt1_3rdDim: $filt1_3rdDim\nchan2_n: $chan2_n\nfilt2_size: $filt2_size\nfilt2_3rdDim: $filt2_3rdDim\nchan3_n: $chan3_n\nfilt3_size: $filt3_size\nfilt3_3rdDim: $filt3_3rdDim\nBatchNorm=$BatchNorm\nMaxPool=$MaxPool\nc_trial: $c_trial" > $LOG_DIR/$JOB_ID-out.txt

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%s\t' $JOB_ID $expDate $mdl_name $path_model_save_base $fname_data_train_val_test $path_existing_mdl $thresh_rr $temporal_width $pr_temporal_width $bz_ms $nb_epochs $chan1_n $filt1_size $filt1_3rdDim $chan2_n $filt2_size $filt2_3rdDim $chan3_n $filt3_size $filt3_3rdDim $BatchNorm $MaxPool $c_trial $use_chunker $TRSAMPS $lr | paste -sd '\t' >> job_list.csv
 
 done
 
done
