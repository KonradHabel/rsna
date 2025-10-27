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
    model:              str   = 'convnext_base.dinov3_lvd1689m'
    classifier_dropout: float = 0.1 
    gc:                 bool  = False          # use gradient checkpointing
    
    # Dataset
    fold:               int   = 0 
    train_on_all:       bool  = False
    steps:              tuple = (1,2)
    eval_step:          int   = 1
    oversample:         int   = 1
    random_shift:       int   = 1
    random_shift_p:     float = 0.5
    random_hard_p:      float = 0.5
    random_flip_p:      float = 0.5
    random_drift:       float = 0.01
    
    # Transforms
    img_size:           tuple = (448, 448)     # crop size         
    mean:               float = 2.5
    std:                float = None

    # Training 
    seed:               int   = 44             # seed for Python, Numpy, Pytorch
    epochs:             int   = 20             # epochs to train
    batch_size:         int   = 32             # batch size for training
    gpu_ids:            tuple = (0,1)          # GPU ids for training e.g. (0,1) multi GPU     


    # Learning Rate
    lr:                 float = 0.00003                      
    warmup_epochs:      int   = 1.0
    scheduler:          str   = "constant"     # "polynomial" | "cosine" | "constant" | None
    lr_end:             float = 0.000005       # only for "polynomial"

    
    # Optimizer  
    gradient_clipping:  bool  = True         
    use_ema:            bool  = True
    ema_step:           int   = 3
    ema_decay:          float = 0.999 

    # Loss
    focal:              bool  = True           # True: Focal Loss | False: BCE Loss
    alpha:              float = None
    gamma:              float = 2.0
    
    # Eval
    zero_shot:          bool = True            # eval before training
    
    # Save path for model checkpoints
    model_path:         str  = "./model"          
    
    # Checkpoint to start from
    checkpoint_start: str = None     
    
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

model_path = "{}/{}/fold-{}".format(config.model_path,
                                       config.model,
                                       config.fold)

if not os.path.exists(model_path):
    os.makedirs(model_path)
    
shutil.copyfile(os.path.basename(__file__), "{}/train_stage1.py".format(model_path))

# Redirect print to both console and log file
sys.stdout = Logger("{}/log_stage1.txt".format(model_path))

# Set seed
setup_system(seed=config.seed,
             cudnn_benchmark=config.cudnn_benchmark,
             cudnn_deterministic=config.cudnn_deterministic)

#----------------------------------------------------------------------------------------------------------------------#  
# Model                                                                                                                #
#----------------------------------------------------------------------------------------------------------------------#  

print("\nModel: {}".format(config.model))
 
model = Net(encoder_name=config.model,
            classifier_dropout=config.classifier_dropout,
            gc=config.gc)

# load pretrained Checkpoint    
if config.checkpoint_start is not None:  
    print("\nStart from:", config.checkpoint_start)
    model_state_dict = torch.load(config.checkpoint_start)  
    model.load_state_dict(model_state_dict, strict=True)
    
# Data parallel
print("\nGPUs available:", torch.cuda.device_count())  
if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
    print("Using Data Prallel with GPU IDs: {}".format(config.gpu_ids))
    model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)    

# Model to device   
model = model.to(config.device)


#----------------------------------------------------------------------------------------------------------------------#  
# DataLoader                                                                                                           #
#----------------------------------------------------------------------------------------------------------------------#  

print(f"\nMean: {config.mean} - Std: {config.std}\n")

df = pd.read_csv("./data/train_stage1_10_folds.csv")

df_train = df[df["fold"]!=config.fold].copy().reset_index(drop=True)

if config.train_on_all:
    df_train = df

if config.oversample > 0:
    df_list = [df_train[df_train["class_id"]!=13]] * config.oversample + [df_train]
    df_train = pd.concat(df_list).reset_index(drop=True)
      
df_valid = df[(df["fold"]==config.fold) & (df["f"]!=-1)].copy().reset_index(drop=True)

with open("./data/f_dict_sample.pkl", "rb") as f:
    f_dict_sample = pickle.load(f)    

with open("./data/label_dict.pkl", "rb") as f:
    label_dict = pickle.load(f) 


if os.path.isfile("./data/f_dict_hard.pkl"):
    with open("./data/f_dict_hard.pkl", "rb") as f:
        f_dict_hard = pickle.load(f) 
else:
    print("Dictionary for sampling of hard negativ spots not found.")
    f_dict_hard = None
    

# Transforms
valid_transforms = A.Compose([
                              A.CenterCrop(config.img_size[0], config.img_size[1]),
                              ]) 


train_transforms = A.Compose([
                              A.ShiftScaleRotate(rotate_limit=(-5, 5), p=0.5),
                              A.RandomCrop(config.img_size[0], config.img_size[1]),
                              A.RandomRotate90(p=1.0),
                              A.OneOf([
                                        A.GridDropout(ratio=0.4, p=0.5),
                                        A.CoarseDropout(max_holes=25,
                                                        max_height=int(0.2*config.img_size[0]),
                                                        max_width=int(0.2*config.img_size[0]),
                                                        min_holes=10,
                                                        min_height=int(0.1*config.img_size[0]),
                                                        min_width=int(0.1*config.img_size[0]),
                                                        p=0.5),
                                        A.GridDistortion(p=1.0),
                                        ], p=0.5),
                              ]) 


# Train                                    
train_dataset = TrainDataset(df=df_train,
                             folder="./data/npy_slice",
                             transforms=train_transforms,
                             steps=config.steps,
                             random_shift=config.random_shift,
                             random_shift_p=config.random_shift_p,
                             random_flip_p=config.random_flip_p,
                             f_dict_sample=f_dict_sample, 
                             label_dict=label_dict, 
                             mean=config.mean,
                             std=config.std,
                             random_drift=config.random_drift,
                             f_dict_hard=f_dict_hard,
                             random_hard_p=config.random_hard_p)      

                          
train_loader = DataLoader(train_dataset,
                          batch_size=config.batch_size,
                          num_workers=config.num_workers,
                          shuffle=True,
                          pin_memory=True,
                          drop_last=True,
                          )
                          
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
# Loss                                                                                                                 #
#----------------------------------------------------------------------------------------------------------------------#  

if config.focal:
    loss_function = FocalLoss(mode="multilabel",
                              alpha=config.alpha,
                              gamma=config.gamma,
                              ignore_index=-1,
                              normalized=False)
    
    print("Using focal loss")
    
else:
    
    loss_function = torch.nn.BCEWithLogitsLoss()
                                                                                                                                                                                                                                                                                            
#----------------------------------------------------------------------------------------------------------------------#  
# Optimizer and scaler                                                                                                 #
#----------------------------------------------------------------------------------------------------------------------#  

optimizer =  Mars(model.parameters(), lr=config.lr, weight_decay=0.01)

if not torch.cuda.is_available():
    print("No GPU available using float32 on CPU.")
    scaler = None
    autocast_dtype = torch.float32
    
else:

    # Check for native bfloat16 support
    if torch.cuda.is_bf16_supported(including_emulation=False):
        print("Using bfloat16 (native support)")
        scaler = None
        autocast_dtype = torch.bfloat16
                    
    # Fall back to float16 with gradient scaling
    elif torch.cuda.get_device_capability()[0] >= 7:  
        print("Using float16 with gradient scaling")
        scaler = torch.cuda.amp.GradScaler()
        autocast_dtype = torch.float16

    else:
        print("Using float32 (GPU too old for mixed precision)")
        scaler = None
        autocast_dtype = torch.float32 
    
#----------------------------------------------------------------------------------------------------------------------#  
# Scheduler                                                                                                            #
#----------------------------------------------------------------------------------------------------------------------#  
if config.scheduler is not None:
    scheduler = get_scheduler(config,
                              optimizer,
                              train_loader_length=len(train_loader),
                              power=1.5)       
else:
    scheduler = None
    
    
#----------------------------------------------------------------------------------------------------------------------#  
# Gradient clipping and EMA                                                                                            #
#----------------------------------------------------------------------------------------------------------------------#     
    
grad_history = RingBuffer(capacity=256, dtype=np.float32)   


if config.use_ema:
    ema = ExponentialMovingAverage(model.parameters(), decay=config.ema_decay) 
else:
    ema = None    
    

#----------------------------------------------------------------------------------------------------------------------#  
# Zero Shot                                                                                                            #
#----------------------------------------------------------------------------------------------------------------------# 

if config.zero_shot:
     
    print_line(name="Evaluate", length=40)
        
    probs, labels = predict(config, 
                            model,
                            dataloader=val_loader,
                            ema=ema,
                            autocast_dtype=autocast_dtype)
    
    score, scores = weighted_multilabel_auc(labels, probs, weights)

    df_score =  pd.DataFrame({"names": classes + ["Mean"],
                              "roc": scores.tolist() + [score]})

    print(df_score.to_string(index=False))
    
        

#----------------------------------------------------------------------------------------------------------------------#  
# Train                                                                                                                #
#----------------------------------------------------------------------------------------------------------------------#  

best_score = 0

for epoch in range(1, config.epochs+1):
    
    print_line(name="Epoch: {}".format(epoch), length=80)
    
    # Train
    train_loss = train(config,
                       model,
                       dataloader=train_loader,
                       loss_function=loss_function,
                       optimizer=optimizer,
                       ema=ema,
                       scheduler=scheduler,
                       scaler=scaler,
                       autocast_dtype=autocast_dtype,
                       grad_history=grad_history,
                       )

    print("Avg. Train Loss = {:.4f} - Lr = {:.6f}\n".format(train_loss,
                                                           optimizer.param_groups[0]['lr']))

    # Evaluate
    print_line(name="Evaluate", length=40)
        
    probs, labels = predict(config, 
                            model,
                            dataloader=val_loader,
                            ema=ema,
                            autocast_dtype=autocast_dtype)
    
    score, scores = weighted_multilabel_auc(labels, probs, weights)

    df_score =  pd.DataFrame({"names": classes + ["Mean"],
                              "roc": scores.tolist() + [score]})

    print(df_score.to_string(index=False))
    

       
    # Save weights   
    checkpoint_path = '{}/weights_e{}_{:.4f}.pth'.format(model_path,
                                                         epoch,
                                                         score)

    if ema is not None:
        with ema.average_parameters():  
            if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
                torch.save(model.module.state_dict(), checkpoint_path)
            else:
                torch.save(model.state_dict(), checkpoint_path)
    else: 
        if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
            torch.save(model.module.state_dict(), checkpoint_path)
        else:
            torch.save(model.state_dict(), checkpoint_path)  
                    
    if score > best_score:
        shutil.copy(checkpoint_path, f'{model_path}/best_stage1.pth') 
        
        
        
    
