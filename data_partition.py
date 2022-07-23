import os
from multiprocessing import Pool
import shutil
import argparse
from sklearn.model_selection import train_test_split

def gen_roots(_file):
    delim_ind = _file.rfind("_")
    prefix = _file[0:delim_ind]
    return(prefix)

def move(_file_list,_file_parents,_root_data_dir,_target_dir,_current_path):
    _data_dir = os.path.join(_root_data_dir, _target_dir)
    if not os.path.isdir(_data_dir):
        os.makedirs(_data_dir)
    _file_parents = set(_file_parents)
    for _file in _file_list:
        prefix = gen_roots(_file)
        if prefix in _file_parents:
            shutil.move(os.path.join(_current_path,_file), _data_dir)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="kspace to image space conversion for single and multi coil MRI scans")

    #####################
    #Parse CLI arguments
    #####################
    parser.add_argument(
    '--source',
    '-s',
    required=True,
    nargs = 1,
    type=str,
    help="Specify a source file path"
    )

    parser.add_argument(
    '--target',
    '-t',
    required=True,
    nargs = 1,
    type=str,
    help="Specify a destination file path"
    )

    parser.add_argument(
    '-a',
    default=False,
    action='store_true',
    help="The -a flag indicates distribution A."
    )

    parser.add_argument(
    '-b',
    default=False,
    action='store_true',
    help="The -b flag indicates distribution B."
    )

    args = parser.parse_args()
    #Check for conflicting options
    if (args.a and args.b) or (not args.a and not args.b):
        print("Exactly one of the -a or -b flags must be chosen\n")
        exit()

    #Set up variables
    file_path = args.source[0]
    target = args.target[0]

    if args.a:
        distribution = "A"
    elif args.b:
        distribution = "B"

    train_dir = "train"+distribution
    test_dir = "test"+distribution
    val_dir = "val"+distribution

    file_names = os.listdir(file_path)

    with Pool(6) as p:
       parent_files = list(set(p.map(gen_roots,file_names)))

    train, tes_tval = train_test_split(parent_files, test_size=0.40, shuffle=True,random_state=1)
    test, val = train_test_split(tes_tval, test_size=0.5, shuffle=True, random_state=1)
    
    move(file_names,train,target,train_dir,file_path)
    print("Training files moved.\n")
    move(file_names,test,target,test_dir,file_path)
    print("Testing files moved.\n")
    move(file_names,val,target,val_dir,file_path)
    print("Validation files moved.\n")

    
    

