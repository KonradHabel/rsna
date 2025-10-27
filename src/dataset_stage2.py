import numpy as np
from torch.utils.data import Dataset
import torch
 

class TrainDataset(Dataset):
    def __init__(self,
                 series,
                 features,
                 probs,
                 labels,
                 cut=256,
                 step=1,
                 train=False,
                 ):
        

        self.series = series
        self.features = features
        self.labels = labels
        self.cut = cut
        self.train = train
        self.step = step
        self.probs = probs

        
    def __getitem__(self, index):
        
        
        uid = self.series[index]
        
        if self.train:
            aug = np.random.choice(self.step) - 1
        else:
            aug = self.step - 1
        
                
        features = self.features[aug][uid].cpu()
        
        labels_cls = torch.tensor(self.labels[index], dtype=torch.float32)
   
        length = len(features)
        
        if self.cut is not None:
            
            cut = self.cut
                
            step = int(length / cut)
                
            if step > 1:
                
                if self.train:
                    step_select = np.random.choice(np.arange(step))
                    step = max(1, step_select)
                    start = np.random.choice(np.arange(step))
                else:
                    start = 0
                     
                features = features[start::step]
        
            length = len(features)
        
        pad_size_features    = 2048 - length
        pad_size_seq         = 2048 - length
        
        att_mask    = [1] * len(features) + [0] * pad_size_seq

        features_pad = torch.zeros((pad_size_features, features.shape[-1]), dtype=features.dtype)
        
        features = torch.cat([features, features_pad], dim=0)
        attention_mask = torch.tensor(att_mask, dtype=torch.float)
        
        return features, attention_mask, labels_cls, length#, view
    
    def __len__(self):

        return len(self.series)
    
    def smart_batching_collate(self, batch):
        
        # max sequence length in batch without padding
        length = [x[3] for x in batch]
        length = torch.tensor(length, dtype=torch.int64)  
        max_lenght = min(1024, length.max())
        
        
        # stack and cut to max sequence length in batch
        input_ids = [x[0] for x in batch]
        input_ids = torch.stack(input_ids)
        input_ids = input_ids[:,:max_lenght]
        
        attention_mask = [x[1] for x in batch]
        attention_mask = torch.stack(attention_mask)
        attention_mask = attention_mask[:,:max_lenght]
        

        label_cls = [x[2] for x in batch]
        label_cls = torch.stack(label_cls, 0) 
        

        return input_ids, attention_mask, label_cls
        
        

    
