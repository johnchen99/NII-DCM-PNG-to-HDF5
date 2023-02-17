import os
import glob
import h5py
import cv2
import numpy as np
import pydicom
from natsort import natsorted
from PIL import Image
import SimpleITK as sitk  
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut

# Function to convert DICOM files to HDF5 format
def dicom_to_hdf5(patient, path):
    dicom_list = [] 
    
    # Loop over all DICOM files in the specified directory and append their pixel arrays to a list
    for dir_image in natsorted(os.listdir(path)):
        img = pydicom.dcmread(os.path.join(path, dir_image))
        arr = img.pixel_array
        arr = apply_modality_lut(arr, img) # Apply the modality lookup table to the pixel array
        arr = apply_voi_lut(arr, img, index=0) # Apply the VOI lookup table to the pixel array
        dicom_list.append(arr)
        print(f'Patient ' + patient + ', Processed DICOM: ' + dir_image)

    img_np = np.array(dicom_list, dtype=object)  
    
    # Create a new HDF5 file and store the DICOM pixel arrays in a dataset under a subgroup named after the patient
    f = h5py.File(h5_main_path, 'a') 
    f.create_dataset(name=patient+'/dicom', data=img_np.astype(np.int16), chunks=True, compression='gzip', compression_opts=9, shuffle=True)
    f.close() 
    
# Function to convert PNG files to HDF5 format
def png_to_hdf5(patient, path):
    png_list = [] 
    
    # Loop over all PNG files in the specified directory and append their pixel arrays to a list
    for dir_image in natsorted(os.listdir(path)):
        img = cv2.imread(os.path.join(path, dir_image))
        png_list.append(img)  
        print(f'Patient ' + patient + ', Processed PNG: ' + dir_image)

    img_np = np.array(png_list, dtype=object)  
    
    # Create a new HDF5 file and store the PNG pixel arrays in a dataset under a subgroup named after the patient
    f = h5py.File(h5_main_path, 'a') 
    f.create_dataset(name=patient+'/ground', data=img_np.astype(int), chunks=True, compression='gzip', compression_opts=9, shuffle=True)  
    f.close()  

# Paths 
train_root = 'D:/CHAOS/CHAOS_Train_Sets/Train_Sets'
h5_filename = 'CT_train_compress.h5'
h5_main_path = train_root + '/' + h5_filename
patient_list = natsorted(os.listdir(train_root+'/CT'))

# Loop over all patients in the patient list
if os.path.exists(h5_main_path):
    print("H5 file already exists: " + h5_main_path)
else:
    # Create a new HDF5 file with a group for each patient and store the DICOM and PNG data for each patient in their respective subgroups
    with h5py.File(h5_main_path, 'w', track_order=True) as f:
        for patient in patient_list:
            ground_data_path = train_root + '/CT/' + patient + '/Ground'
            dicom_data_path = train_root + '/CT/' + patient + '/DICOM_anon'

            # Create subgroup for each patient
            f.create_group(patient)
            
            # Check file counts
            png_file_count = len(glob.glob(ground_data_path+'/*.png'))
            dicom_file_count= len(glob.glob(dicom_data_path+'/*.dcm'))
            print(f'Patient {str(patient)}, Reading ground dir: {ground_data_path}; total ground files={png_file_count}')
            print(f'Patient {str(patient)}, Reading dicom dir: {dicom_data_path}; total dicom files={dicom_file_count}')
            
            png_to_hdf5(patient,ground_data_path)
            dicom_to_hdf5(patient,dicom_data_path)
            print("Patient "+str(patient)+", Done!")
###########################################################################################
# # Test Conversion
# # Test .h5 to .png
# h5 = h5py.File(h5_png_path,'r')
# array = h5["ground"][:]
# img1 = Image.fromarray(array[50].astype('uint8'), 'RGB')
# img1.save("test", "PNG")
# mg1.show()

# # Test .h5 to .dcm
# h5 = h5py.File(h5_dcm_path,'r')
# array = h5["dicom"][:]
# # Inverse bit
# img2 = sitk.GetImageFromArray(1-array[50].astype(np.int16))
# sitk.WriteImage(img2, "D:/CHAOS/CHAOS_Train_Sets/Train_Sets/CT/1/testdcm.dcm")



