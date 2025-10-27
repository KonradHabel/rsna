import os
import cv2

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
cv2.setNumThreads(2)

import torch
import pickle
import pandas as pd
from dataclasses import dataclass
from src.model import Stage1 as Net
import numpy as np
from torch.amp import autocast
from tqdm import tqdm
import math

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

    fold:         int   = 0
    
    crop_size:    int   = 448

    model:        str   = 'convnext_base.dinov3_lvd1689m'
    batch_size:   int   = 256 

    mean:         float = 2.5
    std:          float = None
    
    model_path:   str  = "./model" 
    
    # GPU ids for extraction e.g. (0,1) multi GPU or (0,) for single GPU
    gpu_ids:      tuple = (0,1) 
    
    # show progress bar
    verbose:      bool = True 
  
    # set num_workers to 0 on Windows
    num_workers:  int  = 0 if os.name == 'nt' else 4  
    
    # train on GPU if available
    device:       str = 'cuda' if torch.cuda.is_available() else 'cpu' 
    

#----------------------------------------------------------------------------------------------------------------------#  
# Config                                                                                                               #
#----------------------------------------------------------------------------------------------------------------------#  
config = Configuration()

config.crop = int((512 - config.crop_size) / 2)


model_path = f"{config.model_path}/{config.model}/fold-{config.fold}"
weights_path = f"{model_path}/best_stage1.pth"

df = pd.read_csv("./data/train.csv")

#----------------------------------------------------------------------------------------------------------------------#  
# Model                                                                                                                #
#----------------------------------------------------------------------------------------------------------------------#  
print("\nModel: {}".format(config.model))

model = Net(encoder_name=config.model,
            classifier_dropout=0.0,
            gc=False)
     
# load pretrained Checkpoint    
print("\nStart from:", weights_path)
model_state_dict = torch.load(weights_path)         
model.load_state_dict(model_state_dict, strict=True) 
  

# Data parallel
print("\nGPUs available:", torch.cuda.device_count())  
if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
    print("Using Data Prallel with GPU IDs: {}".format(config.gpu_ids))
    model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)    

  
# Model to device   
model = model.to(config.device)

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
                    
    # Fall back to float16 with gradient scaling
    elif torch.cuda.get_device_capability()[0] >= 7:  
        print("Using float16 with gradient scaling")
        autocast_dtype = torch.float16

    else:
        print("Using float32 (GPU too old for mixed precision)")
        autocast_dtype = torch.float32 


#----------------------------------------------------------------------------------------------------------------------#  
# Extract                                                                                                              #
#----------------------------------------------------------------------------------------------------------------------#

print(f"\nCenter-Crop from 512x512 -> {512-config.crop}x{512-config.crop}\n")

probs_dict_list = list()
features_dict_list = list()
logits_dict_list = list()

for j in range(2):
    
    probs_dict_list.append(dict())
    features_dict_list.append(dict())
    logits_dict_list.append(dict())


for i in tqdm(range(len(df))):
    
    
    row = df.iloc[i]
    
    series = row['SeriesInstanceUID']
    
    data = np.load(f"./data/npy_sequence/{series}.npy")
    
    images = data[:, config.crop:-config.crop, config.crop:-config.crop]
    
    images = images - config.mean
    
    if config.std is not None:
        images = images / config.std
        
        
    images = images.astype(np.float32)    
    images = torch.from_numpy(images)
    
 
    left2 = torch.roll(images, 2, 0)
    left1 = torch.roll(images, 1, 0)
    
    right1 = torch.roll(images, -1, 0)
    right2 = torch.roll(images, -2, 0)    
    
    images1 = torch.stack([left1, images, right1], dim=1)
    images2 = torch.stack([left2, images, right2], dim=1)

    
    batches = math.ceil(len(images) / config.batch_size)
    
    probs1_list = []
    features1_list = []
    logits1_list = []
    
    probs2_list = []
    features2_list = []
    logits2_list = []

    for b in range(batches):
        
        start = b * config.batch_size
        end = start + config.batch_size
    
    
        batch1 = images1[start:end].clone()
        batch2 = images2[start:end].clone()

        with autocast(device_type=config.device, dtype=autocast_dtype), torch.no_grad():
            
            #print(batch.shape)
    
            batch1 = batch1.to(config.device)
            batch2 = batch2.to(config.device)
            
            # Forward pass
            logits1, features1 = model(batch1)
            logits2, features2 = model(batch2)    
    
            probs1 = logits1.sigmoid().cpu()
            logits1, features1 = logits1.cpu(), features1.cpu()
            
            probs2 = logits2.sigmoid().cpu()
            logits2, features2 = logits2.cpu(), features2.cpu()
            
            
            probs1_list.append(probs1)
            logits1_list.append(logits1)
            features1_list.append(features1.cpu())
            
            
            probs2_list.append(probs2)
            logits2_list.append(logits2)
            features2_list.append(features2.cpu())
            
        
    if batches == 1:
        probs1 = probs1_list[0]
        features1 = features1_list[0]
        logits1 = logits1_list[0]
        
        probs2 = probs2_list[0]
        features2 = features2_list[0]
        logits2 = logits2_list[0]
        
    else:
        probs1 = torch.cat(probs1_list)
        features1 = torch.cat(features1_list)
        logits1 = torch.cat(logits1_list) 

        probs2 = torch.cat(probs2_list)
        features2 = torch.cat(features2_list)
        logits2 = torch.cat(logits2_list)  

    probs_dict_list[0][series] = probs1.to(torch.float32)
    features_dict_list[0][series] = features1.to(torch.float32)
    logits_dict_list[0][series] = logits1.to(torch.float32)
    
    
    probs_dict_list[1][series] = probs2.to(torch.float32)
    features_dict_list[1][series] = features2.to(torch.float32)
    logits_dict_list[1][series] = logits2.to(torch.float32)


#----------------------------------------------------------------------------------------------------------------------#  
# Save                                                                                                                 #
#----------------------------------------------------------------------------------------------------------------------#

with open(f"{model_path}/probs_dict.pkl", "wb") as f:
    pickle.dump(probs_dict_list, f)

with open(f"{model_path}/features_dict.pkl", "wb") as f:
    pickle.dump(features_dict_list, f)
    
with open(f"{model_path}/logits_dict.pkl", "wb") as f:
    pickle.dump(logits_dict_list, f)    