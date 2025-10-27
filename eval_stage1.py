import os
import cv2
import pandas as pd

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
cv2.setNumThreads(2)
pd.options.display.float_format = '{:.4f}'.format

import sys
import shutil
import torch
import pickle
import numpy as np

from dataclasses import dataclass
from torch.utils.data import DataLoader

import albumentations as A
from timm.optim.mars import Mars
from numpy_ringbuffer import RingBuffer
from torch_ema import ExponentialMovingAverage

from src.model import Stage1 as Net
from src.dataset_stage1 import ValidDataset, TrainDataset
from src.trainer_stage1 import train
from src.scheduler import get_scheduler
from src.evaluator_stage1 import predict
from src.utils import Logger, setup_system, print_line

from src.focal_loss import FocalLoss
from src.metric import weighted_multilabel_auc


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
    
    # Dataset
    fold:               int   = 0 
    eval_step:          int   = 1

    # Transforms
    img_size:           tuple = (448, 448)     # crop size         
    mean:               float = 2.5
    std:                float = None

    # Training 
    batch_size:         int   = 32
    
    # Save path for model checkpoints
    model_path:         str  = "./model" 
         
    # show progress bar
    verbose: bool = True 
  
    # set num_workers to 0 on Windows
    num_workers: int = 0 if os.name == 'nt' else 4  
    
    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu' 
    
    # for better performance
    cudnn_benchmark: bool = True
    
    # make cudnn deterministic
    cudnn_deterministic: bool = False     
    

#----------------------------------------------------------------------------------------------------------------------#  
# Config                                                                                                               #
#----------------------------------------------------------------------------------------------------------------------#  
config = Configuration()

model_path = f"{config.model_path}/{config.encoder_name}/fold-{config.fold}"

#----------------------------------------------------------------------------------------------------------------------#  
# Model                                                                                                                #
#----------------------------------------------------------------------------------------------------------------------#  

print("\nModel: {}".format(config.encoder_name))
 
model = Net(encoder_name=config.encoder_name,
            classifier_dropout=0.0,
            gc=False)

# load pretrained Checkpoint    
checkpoint_path_inference = '{}/best_stage1.pth'.format(model_path)
print("\nLoad Checkpoint:", checkpoint_path_inference)
model_state_dict = torch.load(checkpoint_path_inference)  
model.load_state_dict(model_state_dict, strict=True)
    
# Model to device   
model = model.to(config.device)

#----------------------------------------------------------------------------------------------------------------------#  
# DataLoader                                                                                                           #
#----------------------------------------------------------------------------------------------------------------------#  

print(f"\nMean: {config.mean} - Std: {config.std}\n")

df = pd.read_csv("./data/train_stage1_10_folds.csv")

      
df_valid = df[(df["fold"]==config.fold) & (df["f"]!=-1)].copy().reset_index(drop=True)


with open("./data/label_dict.pkl", "rb") as f:
    label_dict = pickle.load(f) 

# Transforms
valid_transforms = A.Compose([
                              A.CenterCrop(config.img_size[0], config.img_size[1]),
                              ]) 

                          
# Valid                                        
val_dataset = ValidDataset(df=df_valid,
                           folder="./data/npy_slice",
                           transforms=valid_transforms,
                           label_dict=label_dict, 
                           mean=config.mean,
                           std=config.std,
                           step=config.eval_step)

val_loader = DataLoader(val_dataset,
                        batch_size=config.batch_size,
                        num_workers=config.num_workers,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False,
                        )

#----------------------------------------------------------------------------------------------------------------------#  
# Optimizer and scaler                                                                                                 #
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
# Evaluate                                                                                                            #
#----------------------------------------------------------------------------------------------------------------------# 

print_line(name="Evaluate", length=40)
    
probs, labels = predict(config, 
                        model,
                        dataloader=val_loader,
                        ema=None,
                        autocast_dtype=autocast_dtype)

score, scores = weighted_multilabel_auc(labels, probs, weights)

df_score =  pd.DataFrame({"names": classes + ["Mean"],
                          "roc": scores.tolist() + [score]})

print(df_score.to_string(index=False))

        


    
