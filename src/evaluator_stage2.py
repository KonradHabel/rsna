import torch
from torch.amp import autocast
from tqdm import tqdm
import time

@torch.no_grad()
def predict(config, 
            model,
            dataloader,
            ema=None,
            autocast_dtype=torch.float32):
    

    time.sleep(0.5)

    model.eval()

    if config.verbose:
        bar = tqdm(dataloader,
                   total=len(dataloader),
                   ascii=True,
                   bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    else:
        bar = dataloader
        
    probs_list = []
    labels_list = []

    if ema is not None:
        
        with ema.average_parameters():
            
            for features, attention_mask, labels_cls in bar:
                
                features = features.to(config.device) 
                attention_mask = attention_mask.to(config.device) 

                with autocast(device_type=config.device, dtype=autocast_dtype):
    
                    logits = model(features, attention_mask)           
                    probs = logits.sigmoid().cpu()
                    
                    probs_list.append(probs)
                    labels_list.append(labels_cls)

    else:

        for features, attention_mask, labels_cls in bar:
            
            features = features.to(config.device) 
            attention_mask = attention_mask.to(config.device) 
            
            with autocast(device_type=config.device, dtype=autocast_dtype):

                logits = model(features, attention_mask)
                probs = logits.sigmoid().cpu()
                
                probs_list.append(probs)
                labels_list.append(labels_cls)

            
    if config.verbose:
        bar.close()

    probs = torch.cat(probs_list, axis=0).to(torch.float32)
    labels = torch.cat(labels_list, axis=0).to(torch.float32)
    
    return probs, labels




@torch.no_grad()
def predict_inference_transformer(config, 
                                  model,
                                  dataloader,
                                  autocast_dtype=torch.float32):
    
    time.sleep(0.5)

    model.eval()

    if config.verbose:
        bar = tqdm(dataloader,
                   total=len(dataloader),
                   ascii=True,
                   bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    else:
        bar = dataloader
        
    probs_list = []
    labels_list = []

    for features, attention_mask, labels_cls in bar:
        
        features = features.to(config.device) 
        attention_mask = attention_mask.to(config.device) 

        with autocast(device_type=config.device, dtype=autocast_dtype):

            logits = model.forward_transformer(features, attention_mask)
            probs = logits.sigmoid().cpu()
            
            probs_list.append(probs)
            labels_list.append(labels_cls)

        
    if config.verbose:
        bar.close()


    probs = torch.cat(probs_list, axis=0).to(torch.float32)
    labels = torch.cat(labels_list, axis=0).to(torch.float32)
    
    return probs, labels

