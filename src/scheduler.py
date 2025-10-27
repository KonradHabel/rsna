from transformers import get_polynomial_decay_schedule_with_warmup, get_linear_schedule_with_warmup
from transformers import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup

def get_scheduler(train_config, optimizer, train_loader_length, power=2.0):
    
    train_steps = int(train_loader_length * train_config.epochs)
    warmup_steps = int(train_loader_length * train_config.warmup_epochs)
    print("\nWarmup Epochs: {} - Warmup Steps: {}".format(train_config.warmup_epochs, warmup_steps))
    print("Train Epochs:  {} - Train Steps:  {}".format(train_config.epochs, train_steps)) 
       
    if train_config.scheduler == "polynomial":
        print("\nScheduler: polynomial - max LR: {} - end LR: {}".format(train_config.lr, train_config.lr_end))  
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                              num_training_steps=train_steps,
                                                              lr_end = train_config.lr_end,
                                                              power=power,
                                                              num_warmup_steps=warmup_steps)
    elif train_config.scheduler == "cosine":
        print("\nScheduler: cosine - max LR: {}".format(train_config.lr))   

        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_training_steps=train_steps,
                                                    num_warmup_steps=warmup_steps)
        
    elif train_config.scheduler == "linear":
        print("\nScheduler: linear - max LR: {}".format(train_config.lr))
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_training_steps=train_steps,
                                                    num_warmup_steps=warmup_steps)
        
    elif train_config.scheduler == "constant":
        print("\nScheduler: constant - max LR: {}".format(train_config.lr))
        scheduler = get_constant_schedule_with_warmup(optimizer,
                                                      num_warmup_steps=warmup_steps)
        
    else:
        print("\nScheduler: None")
        scheduler = None
        
    return scheduler
           
