###########################################
#Simple program to convert kspace to images
###########################################

import argparse
import h5py
import numpy as np
import fastmri
from fastmri.data import transforms as T
import torchvision.transforms as TT
import sys
from PIL import Image
import os
from multiprocessing import Pool
from matplotlib import pyplot as plt


###########################################
#Functions
###########################################
def delete(_file): #cleanup, cleanup, everybody do your share!
    os.remove(_file)

def single_coil_k2im(_file,_delete): #For conversion of single coil images
    hf = h5py.File(_file, "r")
    volume_kspace = hf['kspace'][()] #get kspace data from h5 file
    num_slices = volume_kspace.shape[0]
    for i in range(num_slices-8):
        slice_kspace = volume_kspace[i+4] #grab slice i
        slice_kspace_tensor = T.to_tensor(slice_kspace) # Convert from numpy array to pytorch tensor
        slice_image = fastmri.ifft2c(slice_kspace_tensor) # Apply Inverse Fourier Transform to get the complex image
        slice_absolute_image = fastmri.complex_abs(slice_image) 
        _im = slice_absolute_image.numpy()
        _im = _im - np.amin(_im)
        _im = (_im/np.max(_im))*255
        _im = Image.fromarray(_im)
        _im = _im.convert("L")
        _im.save(_file[0:-3]+"_"+str(i+1)+".png")
    if _delete:
        delete(_file)

def multi_coil_k2im(_file,_delete): #for conversion of multi coil images
    hf = h5py.File(_file, "r")
    volume_kspace = hf['kspace'][()]
    num_slices = volume_kspace.shape[0]
    num_coils = volume_kspace.shape[1]
    for i in range(num_slices-8):
        slice_kspace = volume_kspace[i+4]
        slice_kspace_tensor = T.to_tensor(slice_kspace)
        slice_image = fastmri.ifft2c(slice_kspace_tensor)
        slice_absolute_image = fastmri.complex_abs(slice_image)
        slice_image_rss = fastmri.rss(slice_absolute_image)
        _im = slice_image_rss.numpy()
        _im = _im - np.amin(_im)
        _im = (_im/np.max(_im))*255
        _im = Image.fromarray(_im)
        _im = _im.convert("L")
        _im.save(_file[0:-3]+"_"+str(i+1)+".png")
    if _delete:
        delete(_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="kspace to image space conversion for single and multi coil MRI scans")
    ###########################################
    #Parser/CLI arguments
    ###########################################
    parser.add_argument(
        '--file-path',
        '-f',
        nargs = 1,
        type=str,
        help="Specify a filepath, ensuring it ends with /"
    )
    parser.add_argument(
        '--single-coil',
        '-s',
        default=False,
        action='store_true',
        help='For single coil conversion.'
    )
    parser.add_argument(
        '--multi-coil',
        '-m',
        default=False,
        action='store_true',
        help="For multi coil conversion."
    )
    parser.add_argument(
        '--delete-h5',
        '-d',
        default=False,
        action='store_true',
        help="Delete the source h5 files after conversion."
    )

    args = parser.parse_args()

    #Check for conflicting options
    if args.single_coil and args.multi_coil:
        print("The -m and -s options cannot be combined. Select either -s or -m.\n")
        exit()

    file_path = args.file_path[0]
    file_names = os.listdir(file_path)    
    file_names = list(filter(lambda f: f[-2:]=="h5", file_names))
    file_list = [os.path.join(file_path,i) for i in file_names]

    delete_opt = [args.delete_h5]*len(file_list)
    #print(delete_opt)
    if args.single_coil:
        with Pool(6) as p:
            p.starmap(single_coil_k2im,list(zip(file_list,delete_opt)))
            #if args.delete_h5:    
                #p.map(delete, file_list)
    elif args.multi_coil:
        with Pool(6) as p:
            p.starmap(multi_coil_k2im, list(zip(file_list,delete_opt)))
            #if args.delete_h5:
                #p.map(delete,file_list)

