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
    transformer_name:   str   = "microsoft/deberta-v3-base"
    hidden_size:        int   = 1024
    intermediate_size:  int   = 1024
    attention_heads:    int   = 1
    num_hidden_layers:  int   = 3
    attention_dropout:  int   = 0.05
    hidden_dropout:     float = 0.15
    classifier_dropout: float = 0.15
    gc:                 bool  = False
    pool:               str   = "gem"    # 'mean', 'gem', 'max'
    
    # Dataset
    fold:               int   = 0
    cut:                int   = 192

    # Training 
    seed:               int   = 42             # seed for Python, Numpy, Pytorch
    epochs:             int   = 12             # epochs to train
    batch_size:         int   = 32             # batch size for training
   
    
    # Learning Rate
    lr:                 float = 0.00001                      
    warmup_epochs:      int   = 0.0
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
    device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu' 
    
    # for better performance
    cudnn_benchmark: bool = True
    
    # make cudnn deterministic
    cudnn_deterministic: bool = False
    

     
#----------------------------------------------------------------------------------------------------------------------#  
# Config                                                                                                               #
#----------------------------------------------------------------------------------------------------------------------#  
config = Configuration()


model_path = f"{config.model_path}/{config.encoder_name}/fold-{config.fold}"

shutil.copyfile(os.path.basename(__file__), "{}/train_stage2.py".format(model_path))

# Redirect print to both console and log file
sys.stdout = Logger("{}/log_stage2.txt".format(model_path))

# Set seed
setup_system(seed=config.seed,
             cudnn_benchmark=config.cudnn_benchmark,
             cudnn_deterministic=config.cudnn_deterministic)

#----------------------------------------------------------------------------------------------------------------------#  
# Model                                                                                                                #
#----------------------------------------------------------------------------------------------------------------------#  

print("\nModel: {}".format(config.transformer_name))

model = Stage2(transformer_name=config.transformer_name,
               hidden_size=config.hidden_size,
               intermediate_size=config.intermediate_size,
               attention_heads=config.attention_heads,
               num_hidden_layers=config.num_hidden_layers,
               attention_dropout=config.attention_dropout,
               hidden_dropout=config.hidden_dropout,
               classifier_dropout=config.classifier_dropout,
               gc=config.gc,
               pool=config.pool)

# Save transformer config
with open('{}/config_stage2.pkl'.format(model_path), "wb") as f:
    pickle.dump(model.config, f)


# Load pretrained Checkpoint    
if config.checkpoint_start is not None:  
    print("\nStart from:", config.checkpoint_start)
    model_state_dict = torch.load(config.checkpoint_start)  
    model.load_state_dict(model_state_dict, strict=True)
    
# Model to device   
model = model.to(config.device)


#----------------------------------------------------------------------------------------------------------------------#  
# DataLoader                                                                                                           #
#----------------------------------------------------------------------------------------------------------------------#  

# Data
df = pd.read_csv("./data/train_10_folds.csv")

df_train = df[df["fold"]!=config.fold]
df_valid = df[df["fold"]==config.fold]


with open(f"{model_path}/features_dict.pkl", "rb") as f:
    features_dict = pickle.load(f)

with open(f"{model_path}/probs_dict.pkl", "rb") as f:
    probs_dict = pickle.load(f)
    
      
# Train                                     
train_dataset = TrainDataset(series=df_train['SeriesInstanceUID'].values.tolist(),
                             features=features_dict,
                             probs=probs_dict,
                             labels=df_train[classes].values,
                             cut=config.cut,
                             train=True,
                             step=[1,2])                             
                             


train_loader = DataLoader(train_dataset,
                          batch_size=config.batch_size,
                          num_workers=config.num_workers,
                          shuffle=True,
                          pin_memory=True,
                          drop_last=True,
                          collate_fn=train_dataset.smart_batching_collate
                          )
                          
# Valid Step 1                                         
val_dataset1 = TrainDataset(series=df_valid['SeriesInstanceUID'].values.tolist(),
                            features=features_dict,
                            probs=probs_dict,
                            labels=df_valid[classes].values,
                            cut=config.cut,
                            train=False,
                            step=1)                             
                             

val_loader1 = DataLoader(val_dataset1,
                         batch_size=config.batch_size,
                         num_workers=config.num_workers,
                         shuffle=False,
                         pin_memory=True,
                         drop_last=False,
                         collate_fn=val_dataset1.smart_batching_collate)

# Valid Step 2   
val_dataset2 = TrainDataset(series=df_valid['SeriesInstanceUID'].values.tolist(),
                            features=features_dict,
                            probs=probs_dict,
                            labels=df_valid[classes].values,
                            cut=config.cut,
                            train=False,
                            step=2)                             
                             


val_loader2 = DataLoader(val_dataset2,
                         batch_size=config.batch_size,
                         num_workers=config.num_workers,
                         shuffle=False,
                         pin_memory=True,
                         drop_last=False,
                         collate_fn=val_dataset2.smart_batching_collate)

#----------------------------------------------------------------------------------------------------------------------#  
# Loss                                                                                                                 #
#----------------------------------------------------------------------------------------------------------------------#  

if config.focal:
    loss_function = FocalLoss(mode='multilabel',
                              alpha=config.alpha,
                              gamma=config.gamma,
                              ignore_index=-1,
                              normalized=True)

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
    
    # Step 1    
    print_line(name="Evaluate Step 1", length=40)
        
    probs, labels = predict(config, 
                            model,
                            dataloader=val_loader1,
                            ema=ema)

    score, scores = weighted_multilabel_auc(labels, probs, weights)

    df_score =  pd.DataFrame({"names": classes + ["Mean"],
                              "roc": scores.tolist() + [score]})

    print(df_score.to_string(index=False))
    
    
    # Step 2 
    print_line(name="Evaluate Step 2", length=40)
    
    probs, labels = predict(config, 
                            model,
                            dataloader=val_loader2,
                            ema=ema)

    score, scores = weighted_multilabel_auc(labels, probs, weights)


    df_score =  pd.DataFrame({"names": classes + ["Mean"],
                              "roc": scores.tolist() + [score]})

    print(df_score.to_string(index=False))    
    
#----------------------------------------------------------------------------------------------------------------------#  
# Train                                                                                                                #
#----------------------------------------------------------------------------------------------------------------------#  

bestscore = 0

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
    print_line(name="Evaluate Step 1", length=40)
        
    
    probs, labels = predict(config, 
                            model,
                            dataloader=val_loader1,
                            ema=ema,
                            autocast_dtype=autocast_dtype)

    score1, scores1 = weighted_multilabel_auc(labels, probs, weights)


    df_score =  pd.DataFrame({"names": classes + ["Mean"],
                              "roc": scores1.tolist() + [score1]})

    print(df_score.to_string(index=False))
    
    
    print_line(name="Evaluate Step 2", length=40)
        
    
    probs, labels = predict(config, 
                            model,
                            dataloader=val_loader2,
                            ema=ema,
                            autocast_dtype=autocast_dtype)

    score2, scores2 = weighted_multilabel_auc(labels, probs, weights)


    df_score =  pd.DataFrame({"names": classes + ["Mean"],
                              "roc": scores2.tolist() + [score2]})

    print(df_score.to_string(index=False))   
    
    
    score = max(score1, score2)
    
           
    if score > bestscore:
        
        bestscore = score
        
        if score1 > score2:
            dataloader_eval = val_loader1
            step_inf = 1
        else:
            dataloader_eval = val_loader2
            step_inf = 2
        

        checkpoint_path = '{}/best_stage2.pth'.format(model_path)

        if ema is not None:
            with ema.average_parameters(): 
                torch.save(model.state_dict(), checkpoint_path)
        else:
            torch.save(model.state_dict(), checkpoint_path)        
                 
        

#----------------------------------------------------------------------------------------------------------------------#  
# Combine Stage I + Stage II                                                                                           #
#----------------------------------------------------------------------------------------------------------------------#

with open('{}/config_stage2.pkl'.format(model_path), "rb") as f:
    transformer_config = pickle.load(f)

# Reload best weights stage1
checkpoint_path_stage1 = '{}/best_stage1.pth'.format(model_path)
model_state_dict_stage1 = torch.load(checkpoint_path_stage1)

# Reload best weights stage2
checkpoint_path_stage2 = '{}/best_stage2.pth'.format(model_path)
model_state_dict_stage2 = torch.load(checkpoint_path_stage2)


state_dict_inference = dict()

for key, value in model_state_dict_stage1.items():
    
    if "image_encoder" in key:
        state_dict_inference[key] = value
        
for key, value in model_state_dict_stage2.items():

    state_dict_inference[key] = value        

# Create inference model
model_inference = Model(encoder_name=config.encoder_name,
                        transformer_config=transformer_config,
                        pool=config.pool)

model_inference.load_state_dict(state_dict_inference, strict=True)

# Save weights for inference model
checkpoint_path_inference = '{}/weights_inference.pth'.format(model_path)
torch.save(model_inference.state_dict(), checkpoint_path_inference)  

model_inference.to(config.device)


# Evaluate inference model on extracted features
print_line(name=f"Evaluate - Step: {step_inf}", length=40)
    
probs, labels = predict_inference_transformer(config, 
                                              model_inference,
                                              dataloader=dataloader_eval,
                                              autocast_dtype=torch.float32)

p = probs.numpy()
label = labels.numpy()

score, scores = weighted_multilabel_auc(labels, p, weights)

df_score =  pd.DataFrame({"names": classes + ["Mean"],
                          "roc": scores.tolist() + [score]})

print(df_score.to_string(index=False))

np.save('{}/probs_stage2.pth'.format(model_path), p)
np.save('{}/labels_stage2.pth'.format(model_path), label)

