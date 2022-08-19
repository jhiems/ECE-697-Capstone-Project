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

#Multi coil train
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



#Knee dicoms
echo "Copying knee_dicoms.tar.gz from staging..."
cp /staging/jhiemstra/knee_dicoms.tar.gz ./
echo "done copying"

echo "Beginning untar"
tar -xzf knee_dicoms.tar.gz -C multicoil --strip-components 1
echo "Done untar"

echo "Removing knee_dicoms.tar.gz tarball"
rm knee_dicoms.tar.gz
echo "Done removing tarball"



# Multi coil val
echo "Copying multicoil_val.tar.gz from staging..."
cp /staging/jhiemstra/multicoil_val.tar.gz ./
echo "done copying"

echo "Beginning untar"
tar -xf multicoil_val.tar.gz -C multicoil --strip-components 1
echo "Done untar"

echo "Removing multicoil_val.tar.gz tarball"
rm multicoil_val.tar.gz
echo "Done removing tarball"



# Multi coil test
echo "Copying knee_multicoil_test_v2.tar.gz from staging..."
cp /staging/jhiemstra/knee_multicoil_test_v2.tar.gz ./
echo "done copying"

echo "Beginning untar"
tar -xf knee_multicoil_test_v2.tar.gz -C multicoil --strip-components 1
echo "Done untar"

echo "Removing knee_multicoil_test_v2.tar.gz tarball"
rm knee_multicoil_test_v2.tar.gz
echo "Done removing tarball"


# Data Partitioning
echo "Multi coil im conversion and data partitioning"
python3 kspace_to_im.py  -f "multicoil" -m -d
python3 data_partition.py -s multicoil -t "data" -a







########################################
# Training
########################################


git clone https://github.com/ccraig3/ece697-mri-denoising.git
mv ece697-mri-denoising/* ./
rm -rf ece697-mri-denoising


python3 train_unet1.py --train_imgs 'data/trainA' --val_imgs 'data/valA' --test_imgs 'data/testA' --train_bias 'fields_5k_poly_train.pkl' --val_bias 'fields_5k_poly_val.pkl' --test_bias 'fields_5k_poly_test.pkl' --ckpt_save_path 'checkpoints' --proj_name 'UNet-L1+L2-cmd-line' --run_name '25 epochs --- wf=5, bicubic' --max_epochs 25 --batch_size 128 --wf 5


python3 train_unet1.py --train_imgs 'data/trainA' --val_imgs 'data/valA' --test_imgs 'data/testA' --train_bias 'fields_5k_poly_train.pkl' --val_bias 'fields_5k_poly_val.pkl' --test_bias 'fields_5k_poly_test.pkl' --ckpt_save_path 'checkpoints' --proj_name 'UNet-L1+L2-cmd-line' --run_name '35 epochs --- wf=6, bicubic' --max_epochs 35 --batch_size 128 --wf 6


python3 train_unet1.py --train_imgs 'data/trainA' --val_imgs 'data/valA' --test_imgs 'data/testA' --train_bias 'fields_5k_poly_train.pkl' --val_bias 'fields_5k_poly_val.pkl' --test_bias 'fields_5k_poly_test.pkl' --ckpt_save_path 'checkpoints' --proj_name 'UNet-L1+L2-cmd-line' --run_name '45 epochs --- wf=7, bicubic' --max_epochs 45 --batch_size 64 --wf 7




tar -czf checkpoints.tar.gz checkpoints
tar -czf test_ims.tar.gz data/testA
echo "DONE"
