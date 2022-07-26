# ECE-697-Capstone-Project


## A collection of code for my ECE 698


## For Peer Review:

Please look at the file ```kspace_to_im.py```. The program takes in our "kspace" data, which is stored as .h5 files 
and runs the data through a series of Fourier transforms and RMS rejoinings. There are separate functions for 
"multicoil" and "single coil" images, as the process to reconstruct the images for these two datatypes differ. I'm 
using argparse in python to make the tool more general, and so I can run things more easily from the CL. Also, note 
that I'm using Multiprocessing to speed things up (since my dataset is quite large)

Unfortunately, I don't have any data for you to try things on :( The data files are GB's large, and github has its 
limits for these file sizes. Thankfully, I know it all runs without bugs because I wind up having to rebuild my data 
frequently!!

If you have any q's, feel free to shoot me a message on teams!

