#!/bin/bash

# verbose
set -x
###################
# Update items below for each train/test
###################

# training params
epochs=100
step=2e-2
wvecDim=25
pretrain=true
dropout=true
# for RNN2 only, otherwise doesnt matter
middleDim=35

model="RNN2" #either RNN, RNN2, RNN3, RNTN, or DCNN


######################################################## 
# Probably a good idea to let items below here be
########################################################
if [ "$model" == "RNN2" ]; then
    outfile="models/${model}_epochs_${epochs}_wvecDim_${wvecDim}_dropout_${dropout}_pretrain_${pretrain}_middleDim_${middleDim}_step_${step}_2.bin"
else
    outfile="models/${model}_epochs_${epochs}_wvecDim_${wvecDim}_dropout_${dropout}_pretrain_${pretrain}_step_${step}_2.bin"
fi


echo $outfile


python runNNet.py --step $step --epochs $epochs --outFile $outfile \
                --middleDim $middleDim --outputDim 5 --wvecDim $wvecDim --model $model --pretrain $pretrain --dropout $dropout 

