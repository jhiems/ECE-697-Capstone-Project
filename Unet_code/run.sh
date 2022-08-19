#!/bin/bash

#Terminate program if anything throws non-zero exit code
set -e

# installation steps for Miniconda
export HOME=$PWD
export PATH
sh Miniconda3-latest-Linux-x86_64.sh -b -p $PWD/miniconda3
export PATH=$PWD/miniconda3/bin:$PATH

########################################
# Environment Creation
########################################
source activate base
conda env create -f environment.yml -n UNET 
conda activate UNET


wandb login `cat wandb_api_key.txt`

########################################
# Preprocessing Pipeline
########################################

#Multi coil
echo "Copying multicoil from staging..."
time cp /staging/jhiemstra/multicoil_train.tar.gz ./
echo "Multicoil done copying"

mkdir multicoil
echo "Beginning untar"
time tar -xf multicoil_train.tar.gz -C multicoil --strip-components 1
echo "Done untar"

echo "Removing multicoil tarball"
time rm multicoil_train.tar.gz
echo "Done removing multicoil tarball"

echo "Multi coil im conversion and data partitioning"
python3 kspace_to_im.py  -f "multicoil" -m -d


########################################
# Training
########################################

python3 Mainfile.py

tar -czf unet_out.tar.gz model_params

echo "DONE"
