import os
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import warnings
from tqdm import tqdm
from multiprocessing import Pool
import dicomsdl as dicom
import pickle
import psutil

warnings.filterwarnings('ignore')

def normalize(img, modality = None):
    """
    Apply normalization
    """
    
    img = img.astype(np.float32)
    
    if modality == 'CT':
        
        p1, p99 = 0, 600
        
        img = np.clip(img, p1, p99)
        
        normalized = (img - p1) / ((p99 - p1) * 0.2)

    else:
        p1, p99 = -1200, 4000
        
        img = np.clip(img, p1, p99)
        
        normalized = (img - p1) / ((p99 - p1) * 0.2)
        
    return normalized
    

def extract(series):  
    
    """
    DICOM -> Numpy
    """

    series_path = f'./data/series/{series}'
    
    series_path = Path(series_path)
    series_name = series_path.name
    
    # Search for DICOM files
    dicom_files = []
    for root, _, files in os.walk(series_path):
        for file in files:
            if file.endswith('.dcm'):
                dicom_files.append(os.path.join(root, file)) 
    
    # Load DICOM files
    if len(dicom_files) == 1:
        
        # Single dicom file 
        ds = dicom.open(dicom_files[0]) 
        info = ds.getPixelDataInfo()
        num_frames = info['NumberOfFrames']
        
        modality = getattr(ds, 'Modality', None)
        slice_uids_sort = [getattr(ds, 'SOPInstanceUID', '')]
        slope = getattr(ds, 'RescaleSlope', None)
        intercept = getattr(ds, 'RescaleIntercept', None)
    
        slices = []   
        
        for i in range(num_frames):
            
            img = ds.pixelData(i)
            
            shape = img.shape
            
            if slope and intercept:
                img = img * slope - intercept
            
            img = normalize(img, modality = modality)
            img = cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_LINEAR_EXACT)
            img = img.astype(np.float16)
            slices.append(img)
            
        slices = np.stack(slices, axis=0)
        
        ds.close()
        
      
    else:
        
        # Multiple dicom files
        slices = []
        slice_uid = []
        slice_position = []
        
        for i, file in enumerate(dicom_files):
            
            ds = dicom.open(file)
            info = ds.getPixelDataInfo()
            num_frames = info['NumberOfFrames']
            
            modality = getattr(ds, 'Modality', None)
            slice_uid.append(getattr(ds, 'SOPInstanceUID', ''))
            position = getattr(ds, 'ImagePositionPatient', None)
            
            if position is None:
                position = float(i)
            else:
                if len(position) >= 3:
                    position = float(position[2])
                else:
                    position = float(getattr(ds, 'InstanceNumber', i))
            
            slice_position.append(position)
            
            if num_frames > 1:
                j = num_frames // 2
            else:
                j = 0
                    
            img = ds.pixelData(j)
             
            shape = img.shape

            slope = getattr(ds, 'RescaleSlope', None)
            intercept = getattr(ds, 'RescaleIntercept', None)
    
            if slope and intercept:
                img = img * slope - intercept
                
            img = normalize(img, modality = modality)
            img = cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_LINEAR_EXACT)
            img = img.astype(np.float16)
            slices.append(img)
            
            ds.close()
            
       
        slices = np.stack(slices, axis=0)
        
        # Sort slices
        idx = zip(list(range(len(slice_position))), slice_position)
        sorted_slices = sorted(idx, key=lambda x: x[1])
        slice_uid = np.array(slice_uid)
        idx_sort = np.array([x[0] for x in sorted_slices])
        slice_uids_sort = slice_uid[idx_sort].tolist()
        slices = slices[idx_sort]
        
        
    # Save slices
    if slices is not None:
  
        save_folder = f"./data/npy_slice/{series}"
        
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            
        for j, s in enumerate(slices):
            np.save(f'{save_folder}/{j:04d}', s)
            
              
    # Save sequence
    np.save(f"./data/npy_sequence/{series}", slices)
            
    l = len(slices)
    h = shape[0]
    w = shape[1]
    
    meta = {'SeriesInstanceUID': series_name,
            "length": l,
            "h": h,
            "w": w,
            "mod": modality}
        
    return series_name, slice_uids_sort, meta
   
   
if __name__ == "__main__":
    
    train_df = pd.read_csv('./data/train.csv')
    
    series_list = train_df['SeriesInstanceUID'].values.tolist()
    
    n_processes = psutil.cpu_count(logical=False)
    
    print(f"Use Cpu-Threads: {n_processes}")
    
    uid_dict = dict()
    meta_list = list()
    
    
    save_folder = "./data/npy_slice"

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    save_folder = "./data/npy_sequence"

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    
    with Pool(processes=n_processes) as pool:
        
        for series, uids, meta in tqdm(pool.imap(extract, series_list), total=len(series_list)):
            uid_dict[series] = uids  
            meta_list.append(meta)
       
    meta_df = pd.DataFrame(meta_list)
           
    with open("./data/slice_uids_dict.pkl", "wb") as f:
        pickle.dump(uid_dict, f)
    
    train_df = train_df.merge(meta_df, on='SeriesInstanceUID', how="left")

    train_df.to_csv("./data/train_meta.csv", index=False)

