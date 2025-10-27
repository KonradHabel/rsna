import os
import sys
import shutil
import torch
import pickle
import numpy as np
import pandas as pd

from dataclasses import dataclass
from torch.utils.data import DataLoader

from timm.optim.mars import Mars
from numpy_ringbuffer import RingBuffer
from torch_ema import ExponentialMovingAverage

from src.model import Stage2, Model
from src.dataset_stage2 import TrainDataset
from src.trainer_stage2 import train
from src.scheduler import get_scheduler
from src.evaluator_stage2 import predict, predict_inference_transformer
from src.utils import Logger, setup_system, print_line

from src.focal_loss import FocalLoss
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
    cut:                int   = 192

    # Training 
    batch_size:         int   = 32       
   
    # Save path for model checkpoints
    model_path:         str  = "./model" 
    
    # show progress bar
    verbose: bool = True 
  
    # set num_workers to 0 on Windows
    num_workers: int = 0 if os.name == 'nt' else 4  
    
    # Device for evaluation
    device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu' 
   
     
#----------------------------------------------------------------------------------------------------------------------#  
# Config                                                                                                               #
#----------------------------------------------------------------------------------------------------------------------#  
config = Configuration()

model_path = f"{config.model_path}/{config.encoder_name}/fold-{config.fold}"

#----------------------------------------------------------------------------------------------------------------------#  
# Model                                                                                                                #
#----------------------------------------------------------------------------------------------------------------------#

# Load config
with open('{}/config_stage2.pkl'.format(model_path), "rb") as f:
    transformer_config = pickle.load(f)


# Create inference model
model_inference = Model(encoder_name=config.encoder_name,
                        transformer_config=transformer_config,
                        pool=config.pool)


# Load weights
checkpoint_path_inference = '{}/weights_inference.pth'.format(model_path)
print(f"\nLoad Checkpoint: {checkpoint_path_inference}\n")
model_state_dict = torch.load(checkpoint_path_inference)
model_inference.load_state_dict(model_state_dict, strict=True)

model_inference.to(config.device)

#----------------------------------------------------------------------------------------------------------------------#  
# DataLoader                                                                                                           #
#----------------------------------------------------------------------------------------------------------------------#  

# Data
df = pd.read_csv("./data/train_10_folds.csv")

df_valid = df[df["fold"]==config.fold]

with open(f"{model_path}/features_dict.pkl", "rb") as f:
    features_dict = pickle.load(f)

with open(f"{model_path}/probs_dict.pkl", "rb") as f:
    probs_dict = pickle.load(f)
    
    
# Valid Step 1                                         
val_dataset = TrainDataset(series=df_valid['SeriesInstanceUID'].values.tolist(),
                           features=features_dict,
                           probs=probs_dict,
                           labels=df_valid[classes].values,
                           cut=config.cut,
                           train=False,
                           step=config.step)                             
                             

val_loader = DataLoader(val_dataset,
                        batch_size=config.batch_size,
                        num_workers=config.num_workers,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False,
                        collate_fn=val_dataset.smart_batching_collate)     

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
# Evaluate                                                                                                             #
#----------------------------------------------------------------------------------------------------------------------# 

print_line(name=f"Evaluate - Step: {config.step}", length=40)
    
probs, labels = predict_inference_transformer(config, 
                                              model_inference,
                                              dataloader=val_loader,
                                              autocast_dtype=autocast_dtype)

p = probs.numpy()
label = labels.numpy()

score, scores = weighted_multilabel_auc(labels, p, weights)

df_score =  pd.DataFrame({"names": classes + ["Mean"],
                          "roc": scores.tolist() + [score]})

print(df_score.to_string(index=False))


