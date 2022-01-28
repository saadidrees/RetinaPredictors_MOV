#!/bin/bash
#int-or-string.sh

EXP_DATE="20180502_s3"

LOG_DIR="/home/sidrees/scratch/dynamic_retina/models/cnn3d/logs/"

if [ ! -d "$LOG_DIR" ]; then 
	mkdir $LOG_DIR
fi


typeset -i runOnCluster=1
typeset -i temporal_width=60

typeset -i chan1_n=13
typeset -i filt1_size=11
typeset -i filt1_3rdDim=25

typeset -i chan2_n=25
typeset -i filt2_size=7
typeset -i filt2_3rdDim=12

typeset -i chan3_n=25
typeset -i filt3_size=3
typeset -i filt3_3rdDim=7

typeset -i nb_epochs=100
thresh_rr='0.6'

typeset -i bz_ms=10000


JOB_ID=$(sbatch --export=LOG_DIR=$LOG_DIR,expDate=$EXP_DATE,runOnCluster=$runOnCluster,chan1_n=$chan1_n,filt1_size=$filt1_size,filt1_3rdDim=$filt1_3rdDim,chan2_n=$chan2_n,filt2_size=$filt2_size,filt2_3rdDim=$filt2_3rdDim,chan3_n=$chan3_n,filt3_size=$filt3_size,filt3_3rdDim=$filt3_3rdDim,nb_epochs=$nb_epochs,thresh_rr=$thresh_rr,temporal_width=$temporal_width,bz_ms=$bz_ms cnn3d_launcher.sh)

echo $JOB_ID
JOB_ID=$(echo "$JOB_ID" | grep -Eo '[0-9]{1,6}')


printf '%s\t%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t' $JOB_ID $EXP_DATE $thresh_rr $temporal_width $bz_ms $nb_epochs $chan1_n $filt1_size $filt1_3rdDim $chan2_n $filt2_size $filt2_3rdDim $chan3_n $filt3_size $filt3_3rdDim| paste -sd '\t' >> job_list.csv
