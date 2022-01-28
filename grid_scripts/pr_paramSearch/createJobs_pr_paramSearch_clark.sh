#!/bin/bash
#int-or-string.sh


# updating repo
cd ../from_git
git fetch --all
git pull
echo "Fetched latest files from git"
cd ../pr_paramSearch

echo $PWD

PARAMS_FILE="pr_paramSearch_params.csv"
LOG_DIR="/home/sidrees/scratch/RetinaPredictors/grid_scripts/pr_paramSearch/logs"

if [ ! -d "$LOG_DIR" ]; then 
	mkdir $LOG_DIR
fi

expDate_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f1) )
path_mdl_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f2) )
trainingDataset_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f3) )
testingDataset_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f4) )
pr_mdl_name_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f5) )
mdl_name_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f6) )
path_excel_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f7) )
path_perFiles_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f8) )
lightLevel_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f9) )
pr_type_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f10) )
samps_shift_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f11) )
c_beta_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f12) )
c_gamma_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f13) )
c_tau_y_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f14) )
c_n_y_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f15) )
c_tau_z_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f16) )
c_n_z_all=( $(tail -n +2 $PARAMS_FILE | cut -d ',' -f17) )


numParams=${#expDate_all[@]}
echo "Number of parameter combinations: $numParams"



#for ((i=0; i<$numParams; i++));
for ((i=0; i<1; i++));
do
 path_mdl=${path_mdl_all[i]}
 trainingDataset=${trainingDataset_all[i]}
 testingDataset=${testingDataset_all[i]}
 mdl_name=${mdl_name_all[i]}
 expDate=${expDate_all[i]}
 path_excel=${path_excel_all[i]}
 path_perFiles=${path_perFiles_all[i]}
 lightLevel=${lightLevel_all[i]}
 pr_type=${pr_type_all[i]}
 samps_shift=${samps_shift_all[i]}
 c_beta=${c_beta_all[i]}
 c_gamma=${c_gamma_all[i]}
 c_tau_y=${c_tau_y_all[i]}
 c_n_y=${c_n_y_all[i]}
 c_tau_z=${c_tau_z_all[i]}
 c_n_z=${c_n_z_all[i]}



 echo "path_mdl: $path_mdl"
 echo "trainingDataset: $trainingDataset"
 echo "testingDataset: $trainingDataset"
 echo "expDate: $expDate"
 echo "pr_mdl_name: $pr_mdl_name"
 echo "mdl_name: $mdl_name"
 echo "path_excel: $path_excel"
 echo "path_perFiles: $path_perFiles"
 echo "lightLevel: $lightLevel"
 echo "pr_type: $pr_type"
 echo "samps_shift: $samps_shift"
 echo "c_beta: $c_beta"
 echo "c_gamma: $c_gamma"
 echo "c_tau_y: $c_tau_y"
 echo "c_n_y: $c_n_y"
 echo "c_tau_z: $c_tau_z"
 echo "c_n_z: $c_n_z"



 JOB_ID=$(sbatch --export=LOG_DIR=$LOG_DIR,path_mdl=$path_mdl,trainingDataset=$trainingDataset,testingDataset=$testingDataset,expDate=$expDate,pr_mdl_name=$pr_mdl_name,mdl_name=$mdl_name,path_excel=$path_excel,path_perFiles=$path_perFiles,lightLevel=$lightLevel,pr_type=$pr_type,samps_shift=$samps_shift,c_beta=$c_beta,c_gamma=$c_gamma,c_tau_y=$c_tau_y,c_n_y=$c_n_y,c_tau_z=$c_tau_z,c_n_z=$c_n_z pr_paramSearch_launcher_clark.sh)

echo $JOB_ID
JOB_ID=$(echo "$JOB_ID" | grep -Eo '[0-9]{1,8}')
 
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t' $JOB_ID $expDate $path_mdl $trainingDataset $testingDataset $pr_mdl_name $mdl_name $path_excel $path_perFiles $lightLevel $pr_type $samps_shift $c_beta $c_gamma $c_tau_y $c_n_y $c_tau_z $c_n_z | paste -sd '\t' >> job_list.csv
 
 
done
