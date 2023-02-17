import os
import glob
import nibabel as nib
import numpy as np
from natsort import natsorted
import h5py

# Convert .png to .h5
def nii_to_hdf5(mode,path):
    nii_list=np.empty(0)
    for nii in natsorted(os.listdir(path+'/'+mode)):
        img = nib.load(path+'/'+mode+'/'+nii).get_fdata()
        arr = np.array(img)
        nii_list = np.append(nii_list, arr)
        print(f'Processed nii: '+nii)

    f = h5py.File(h5_main_path, 'a') 
    if mode == 'ct':
        f.create_dataset(name='ct', data = nii_list,chunks=True, compression='gzip', compression_opts=9, shuffle=True)  
        print(f'Done Processed CT')

    else:
        f.create_dataset(name='label', data = nii_list,chunks=True, compression='gzip', compression_opts=9, shuffle=True)  
        print(f'Done Processed LABEL')
    f.close() 

# Paths 
train_root = '/backdata/raw_dataset/train'
h5_filename = '3DUNET_train_compress.h5'
h5_main_path = train_root+'/'+h5_filename

if os.path.exists(h5_main_path):
    print("H5 file already exist: "+ h5_main_path)
else:
    # Check file counts
    ct_file_count = len(glob.glob(train_root+'/ct/*.nii'))
    label_file_count= len(glob.glob(train_root+'/label/*.nii'))
    print(f'Reading CT dir: {train_root}/ct; total CT files={ct_file_count}')
    print(f'Reading Label dir: {train_root}/label; total Label files={label_file_count}')
    
    nii_to_hdf5('ct',train_root)
    nii_to_hdf5('label',train_root)