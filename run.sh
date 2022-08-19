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
#conda env create -f environment.yml -n MRI
git clone https://github.com/jhiemstrawisc/pytorch-CycleGAN-and-pix2pix.git
source activate base 
cd pytorch-CycleGAN-and-pix2pix
CONDA_RESTORE_FREE_CHANNEL=1 conda env create -f environment.yml -n MRI
conda activate MRI
cd ../


wandb login `cat wandb_api_key.txt`

########################################
# Preprocessing Pipeline
########################################

#Single Coils
echo "Copying single coil from staging..."
time cp /staging/jhiemstra/knee_singlecoil_train.tar.gz ./
echo "Single coil done copying"

mkdir singlecoil

echo "Beginning untar"
time tar -xf knee_singlecoil_train.tar.gz -C singlecoil --strip-components 1
echo "Done untar"

echo "removing single coil tarball"
time rm knee_singlecoil_train.tar.gz
echo "Done removing single coil tarball"

echo "Single coil im conversion and data partitioning"
python3 kspace_to_im.py  -f "singlecoil/" -s -d
python3 data_partition.py -s singlecoil -t "data" -a

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
python3 data_partition.py -s multicoil -t "data" -b


########################################
# Training
########################################

python3 pytorch-CycleGAN-and-pix2pix/train.py --dataroot "data" --name knees --model cycle_gan --batch_size 16 --n_epochs 100 --n_epochs_decay 100 --save_epoch_freq 10 --use_wandb --display_id -1 --no_html --gpu_ids 0,1 --input_nc 1 --output_nc 1 --lr 0.00002

tar -czf model_out.tar.gz pytorch-CycleGAN-and-pix2pix
tar -czf checkpoints_cyclegan_lr_00002.tar.gz checkpoints

echo "DONE"
