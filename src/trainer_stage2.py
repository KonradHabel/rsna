import time
from tqdm import tqdm
from .utils import AverageMeter
from torch.cuda.amp import autocast
import torch
import numpy as np

def _get_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


def train(config,
          model,
          dataloader,
          loss_function,
          optimizer,
          ema=None,
          scheduler=None,
          scaler=None,
          autocast_dtype=torch.float32,
          grad_history=[]):

    model.train()
    
    losses = AverageMeter()
    
    time.sleep(0.5)
    
    optimizer.zero_grad()
    
    step = 1
    
    if config.verbose:
        bar = tqdm(dataloader,
                   total=len(dataloader),
                   ascii=True,
                   bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                   desc="Train")
    else:
        bar = dataloader
        

    for features, attention_mask, labels in bar:
         
        features = features.to(config.device) 
        attention_mask = attention_mask.to(config.device) 
        labels = labels.to(config.device)
        
        if scaler:
            with autocast(device_type=config.device, dtype=autocast_dtype):
                
                logits = model(features, attention_mask)
                
                loss = loss_function(logits, labels)
       
                losses.update(loss.item())
                  
            scaler.scale(loss).backward()
            
            if config.gradient_clipping:
                scaler.unscale_(optimizer)
                
                obs_grad_norm = _get_grad_norm(model)
                grad_history.append(obs_grad_norm)
                
                if len(grad_history) > 50:
                    clip_value = np.percentile(np.array(grad_history), 50)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), min(max(1, clip_value), 100))
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)                
            
            
            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad()
            
            if scheduler is not None:
                scheduler.step()
                    
            if step % config.ema_step == 0 and ema is not None:
                ema.update()
   
        else:

            logits = model(features, attention_mask)
            
            loss = loss_function(logits, labels)
            
            losses.update(loss.item())
            
            loss.backward()
            
            if config.gradient_clipping:
                                
                obs_grad_norm = _get_grad_norm(model)
                grad_history.append(obs_grad_norm)
                
                if len(grad_history) > 50:
                    clip_value = np.percentile(np.array(grad_history), 50)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), min(max(1, clip_value), 100))
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)                
               
            optimizer.step()
            optimizer.zero_grad()
            
            if scheduler is not None:
                scheduler.step()
                    
            if step % config.ema_step == 0 and ema is not None:
                ema.update()
        
        
        if config.verbose:
            monitor = {"loss": "{:.4f}".format(losses.val),
                       "lr" : "{:.6f}".format(optimizer.param_groups[0]['lr'])}
            
            bar.set_postfix(ordered_dict=monitor)
        
        step += 1

    if config.verbose:
        bar.close()
  
    return losses.avg


           
