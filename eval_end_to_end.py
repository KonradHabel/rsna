import os
import torch
import pickle
import math
import numpy as np
import pandas as pd
import dicomsdl as dicom
import cv2
from tqdm import tqdm
from dataclasses import dataclass

from src.model import Model
from src.utils import print_line, setup_system
from src.metric import weighted_multilabel_auc

pd.options.display.float_format = '{:.4f}'.format

# Classes
classes = [
           'Left Infraclinoid Internal Carotid Artery',   # 0
           'Right Infraclinoid Internal Carotid Artery',  # 1
           'Left Supraclinoid Internal Carotid Artery',   # 2
           'Right Supraclinoid Internal Carotid Artery',  # 3
           'Left Middle Cerebral Artery',                 # 4
           'Right Middle Cerebral Artery',                # 5
           'Anterior Communicating Artery',               # 6
           'Left Anterior Cerebral Artery',               # 7
           'Right Anterior Cerebral Artery',              # 8
           'Left Posterior Communicating Artery',         # 9
           'Right Posterior Communicating Artery',        # 10
           'Basilar Tip',                                 # 11
           'Other Posterior Circulation',                 # 12
           'Aneurysm Present',                            # 13
           ]          
           
# Weights for evaluation
weights = np.full(14, 0.5/13, dtype=np.float32)
weights[-1] = 0.5


@dataclass
class Configuration:
    '''
    --------------------------------------------------------------------------
    Models:
    --------------------------------------------------------------------------    
    # 'convnext_base.dinov3_lvd1689m'
    # 'convnext_large.dinov3_lvd1689m'
    
    # 'vit_base_patch16_dinov3.lvd1689m'  
    # 'vit_large_patch16_dinov3.lvd1689m' 
    --------------------------------------------------------------------------
    '''
    
    # Model
    encoder_name:       str   = 'convnext_base.dinov3_lvd1689m'
    pool:               str   = 'gem'
    
    # Eval stepsize 1 or 2
    step:               int   = 1          
    
    # Dataset
    fold:               int   = 0
    crop_size:          int   = 448
    cut:                int   = 192
    mean:               float = 2.5
    std:                float = None 
    batch_size:         int   = 128      
   
    # Save path for model checkpoints
    model_path:         str  = "./model" 
    
    # show progress bar
    verbose: bool = True 
  
    # set num_workers to 0 on Windows
    num_workers: int = 0 if os.name == 'nt' else 4  
    
    # Device for evaluation
    device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu' 
      
def normalize(img, modality = None):
    """
    Apply statistical normalization
    """
    
    img = img.astype(np.float32)
    
    if modality == 'CT':
        
        # Statistical normalization (for CT as well)
        # Normalize using 1-99 percentiles
        p1, p99 = 0, 600
        
        img = np.clip(img, p1, p99)
        
        normalized = (img - p1) / ((p99 - p1) * 0.2)

    else:
        p1, p99 = -1200, 4000
        
        img = np.clip(img, p1, p99)
        
        normalized = (img - p1) / ((p99 - p1) * 0.2)
        
    return normalized    
   
    
def get_data(series_path, config):  
        
    # Search for DICOM files
    dicom_files = []
    for root, _, files in os.walk(series_path):
        for file in files:
            if file.endswith('.dcm'):
                dicom_files.append(os.path.join(root, file))
    
    # Load DICOM datasets
    if len(dicom_files) == 1:
        
        ds = dicom.open(dicom_files[0])
        
        info = ds.getPixelDataInfo()
        num_frames = info['NumberOfFrames']
        
        modality = getattr(ds, 'Modality', None)
        
        slope = getattr(ds, 'RescaleSlope', None)
        intercept = getattr(ds, 'RescaleIntercept', None)
                

        slices = []    
        for i in range(num_frames):
            
            img = ds.pixelData(i)
            
            if slope and intercept:
                img = img * slope - intercept
            
            img = normalize(img, modality = modality)
        
            img = cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_LINEAR_EXACT)
            
            # Center Crop
            img = img[config.crop:-config.crop, config.crop:-config.crop]

            slices.append(img)
            
        images = np.stack(slices, axis=0)
        
        ds.close()
        
        
          
    else:
        
        slices = []
    
        slice_uid = []
        slice_orientation = []
        slice_position = []
        
        
        for i, file in enumerate(dicom_files):
            
            ds = dicom.open(file)
            
            info = ds.getPixelDataInfo()
            num_frames = info['NumberOfFrames']
            
            modality = getattr(ds, 'Modality', None)
            
            
            slice_uid.append(getattr(ds, 'SOPInstanceUID', ''))
            slice_orientation.append(getattr(ds, 'ImageOrientationPatient', None))
            
            
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
            
            slope = getattr(ds, 'RescaleSlope', None)
            intercept = getattr(ds, 'RescaleIntercept', None)

                
            if slope and intercept:
                img = img * slope - intercept
                

            img = normalize(img, modality = modality)
            
            img = cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_LINEAR_EXACT)

            # Center Crop
            img = img[config.crop:-config.crop, config.crop:-config.crop]
            
            slices.append(img)
            
            ds.close()
                
        images = np.stack(slices, axis=0)
        

        idx = zip(list(range(len(slice_position))), slice_position)
        
        
        sorted_slices = sorted(idx, key=lambda x: x[1])
        

        idx_sort = np.array([x[0] for x in sorted_slices])
        

        images = images[idx_sort]


    images = images - config.mean
    
    if config.std is not None:
        images = images / config.std
    
    images = torch.from_numpy(images)

    if config.step == 2:
        left = torch.roll(images, 2, 0)
        right = torch.roll(images, -2, 0)
        
    else:
        left = torch.roll(images, 1, 0)
        right = torch.roll(images, -1, 0)

  
    images = torch.stack([left, images, right], dim=1)


    return images   
    
     
#----------------------------------------------------------------------------------------------------------------------#  
# Config                                                                                                               #
#----------------------------------------------------------------------------------------------------------------------#  
config = Configuration()

config.crop = int((512 - config.crop_size) / 2)


model_path = f"{config.model_path}/{config.encoder_name}/fold-{config.fold}"

#----------------------------------------------------------------------------------------------------------------------#  
# Model                                                                                                                #
#----------------------------------------------------------------------------------------------------------------------#

# Load config
with open('{}/config_stage2.pkl'.format(model_path), "rb") as f:
    transformer_config = pickle.load(f)



# Create inference model
model = Model(encoder_name=config.encoder_name,
              transformer_config=transformer_config,
              pool=config.pool)


# Load weights
checkpoint_path_inference = '{}/weights_inference.pth'.format(model_path)
print("\nLoad Checkpoint:", checkpoint_path_inference)
model_state_dict = torch.load(checkpoint_path_inference)
model.load_state_dict(model_state_dict, strict=True)

model.to(config.device)

model.eval()

#----------------------------------------------------------------------------------------------------------------------#  
# Dtype                                                                                                                #
#----------------------------------------------------------------------------------------------------------------------# 

if not torch.cuda.is_available():
    print("No GPU available using float32 on CPU.")
    autocast_dtype = torch.float32
    
else:

    # Check for native bfloat16 support
    if torch.cuda.is_bf16_supported(including_emulation=False):
        print("Using bfloat16 (native support)")
        autocast_dtype = torch.bfloat16
                    
    # Fall back to float16
    elif torch.cuda.get_device_capability()[0] >= 7:  
        print("Using float16 with gradient scaling")
        autocast_dtype = torch.float16

    else:
        print("Using float32 (GPU too old for mixed precision)")
        autocast_dtype = torch.float32 


#----------------------------------------------------------------------------------------------------------------------#  
# DataLoader                                                                                                           #
#----------------------------------------------------------------------------------------------------------------------#  

# Data
df = pd.read_csv("./data/train_10_folds.csv") 

df_valid = df[df["fold"]==config.fold]

uids = df_valid['SeriesInstanceUID'].values

#----------------------------------------------------------------------------------------------------------------------#  
# Evaluate                                                                                                             #
#----------------------------------------------------------------------------------------------------------------------# 

print(f"\nCenter-Crop from 512x512 -> {512-config.crop}x{512-config.crop}\n")

print_line(name=f"Evaluate - Step: {config.step}", length=40)

probs_list = list()


for uid in tqdm(uids):
    

    series_path = f"./data/series/{uid}"
    images = get_data(series_path, config)
    
    length = len(images)
    step = int(length / config.cut)
    
    if step > 1:
        images = images[0::step]

   
    batches = math.ceil(len(images) / config.batch_size)
     
    features_list = [] 

    for b in range(batches):
                
        start = b * config.batch_size
        end = start + config.batch_size
            
        with torch.amp.autocast(device_type=config.device, dtype=autocast_dtype), torch.no_grad():
                    
            batch = images[start:end].clone().to(torch.device(config.device), non_blocking=True)
                    
            features = model.forward_encoder(batch)
            
        features_list.append(features)
     
    if batches == 1:
        features = features_list[0] 
    else:
        features = torch.cat(features_list)


    features= features.unsqueeze(0)
    mask = torch.ones(1, features.shape[1], dtype=features.dtype, device=features.device)

    with torch.amp.autocast(device_type=config.device, dtype=autocast_dtype), torch.no_grad():    
        logits = model.forward_transformer(features, mask)
        probs = logits[0].sigmoid().cpu()

    probs_list.append(probs)
       
probs = torch.stack(probs_list).to(torch.float32)
                        
p = probs.numpy()
labels = df_valid[classes].values

score, scores = weighted_multilabel_auc(labels, p, weights)

df_score =  pd.DataFrame({"names": classes + ["Mean"],
                          "roc": scores.tolist() + [score]})

print(df_score.to_string(index=False))


