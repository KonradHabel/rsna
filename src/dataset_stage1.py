import numpy as np
from torch.utils.data import Dataset
import torch
import os 

class TrainDataset(Dataset):
    def __init__(self,
                 df,
                 folder,
                 transforms,
                 steps=(1,2),
                 random_shift_p=0.5,
                 random_shift=1,
                 random_flip_p=0.5,
                 f_dict_sample=None,
                 label_dict=None,
                 mean=2.5,
                 std=None,
                 random_drift=0.0,
                 f_dict_hard=None,
                 random_hard_p=0.5):
        

        self.names = df['SeriesInstanceUID'].values
        self.labels = df["location"].values
        self.class_id = df["class_id"].values
                
        self.f_dict_sample = f_dict_sample
        
        self.label_dict = label_dict
        
        self.f = df["f"].values
        
        self.folder = folder
        
        self.length = df["length"].values

        self.transforms = transforms
        
        self.random_drift = random_drift

        self.steps = np.array(steps)
            
        self.random_shift_p = random_shift_p
        self.random_shift = random_shift
        self.random_flip_p = random_flip_p
        
        self.f_dict_hard = f_dict_hard
        self.random_hard_p = random_hard_p

        self.random_shift_choice = np.arange(-random_shift, random_shift+1)
        
        self.mod = df["mod"].values

                
        self.classes = [
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
                       'Nothing',                                     # 13
                       ]

        # Reorder if flip left <> right
        self.reorder = np.array([1,0,3,2,5,4,6,8,7,10,9,11,12,13])
                 
        self.mean = mean
        self.std = std
                      

    def __getitem__(self, index):
        
        
        name = self.names[index]
        class_id = self.class_id[index]
        n = self.length[index]
         
        step = np.random.choice(self.steps)
        
        if class_id == 13:
            i = np.random.choice(self.f_dict_sample[name])

            if self.f_dict_hard is not None and np.random.rand() < self.random_hard_p:
                
                i_list = self.f_dict_hard.get(name, None)
               
                if i_list is not None:
                    i = np.random.choice(i_list)
        
        else:
            
            i = self.f[index]
            
            if np.random.rand() < self.random_shift_p:
                
                shift = np.random.choice(self.random_shift_choice)
                
                i = min(max(0, i + shift), n-1)
                
                
        i = max(0, i)        
   
        image = f"{self.folder}/{name}/{i:04d}.npy"
        
        if not os.path.isfile(image):
            image = f"{self.folder}/{name}/{i-1:04d}.npy"
            
        image = np.load(image)            
           
        image_left = f"{self.folder}/{name}/{i-step:04d}.npy"
            
        if os.path.isfile(image_left):
            image_left = np.load(image_left)            
        else:
            image_left = np.zeros(image.shape, dtype=np.float16)
            
 
        image_right = f"{self.folder}/{name}/{i+step:04d}.npy"
            
        if os.path.isfile(image_right):
            image_right = np.load(image_right)            
        else:
            image_right = np.zeros(image.shape, dtype=np.float16)
            
        
        if np.random.rand() < 0.5:
            image = np.stack([image_left, image, image_right], axis=-1) 
        else:
            image = np.stack([image_right, image, image_left], axis=-1) 
        
        labels = self.label_dict[name][max(0,i-step):min(i+step+1, n-1), :]
        
        if labels.ndim == 2:
            labels = labels.any(0).astype(np.int32)
        
        if np.random.rand() < self.random_flip_p:
            image = np.fliplr(image)
            labels = labels[self.reorder] 
        
        image = self.transforms(image=image.astype(np.float32))['image']
        
        if self.random_drift > 0:
            drift = (np.random.rand() * 2 * self.random_drift) - self.random_drift
            image = image + drift
         
        mean = self.mean
        std = self.std
         
        image = image - mean
        
        if std is not None:
            image = image / std
        
        image = np.moveaxis(image, -1, 0)
        
        image = torch.from_numpy(image.copy())
        labels = torch.from_numpy(labels)
        
        return image, labels
    
    def __len__(self):

        return len(self.names)
    

        
        
class ValidDataset(Dataset):
    def __init__(self,
                 df,
                 folder,
                 transforms,
                 label_dict,
                 mean=2.5,
                 std=None,
                 step=1):
        
    
        self.names = df['SeriesInstanceUID'].values
        self.labels = df["location"].values
        self.class_id = df["class_id"].values
        
        self.mod = df["mod"].values
        
        self.f = df["f"].values
        self.length = df["length"].values
        
        self.folder = folder
        
        self.transforms = transforms

        self.step = step
        
        self.mean = mean
        self.std = std

        self.label_dict = label_dict

    def __getitem__(self, index):
        
        
        name = self.names[index]

        n = self.length[index]
        
        step = self.step
        
        i = self.f[index] 
       
   
        image = f"{self.folder}/{name}/{i:04d}.npy"
        
        if not os.path.isfile(image):
            image = f"{self.folder}/{name}/{i-1:04d}.npy"
            
        image = np.load(image)            
           
 
        image_left = f"{self.folder}/{name}/{i-step:04d}.npy"
            
        if os.path.isfile(image_left):
            image_left = np.load(image_left)            
        else:
            image_left = np.zeros(image.shape, dtype=np.float16)
            

        image_right = f"{self.folder}/{name}/{i+step:04d}.npy"
            
        if os.path.isfile(image_right):
            image_right = np.load(image_right)            
        else:
            image_right = np.zeros(image.shape, dtype=np.float16)
            
        image = np.stack([image_left, image, image_right], axis=-1) 
        
   
        image = self.transforms(image=image.astype(np.float32))['image']
        
        
        mean = self.mean
        std = self.std
         
        image = image - mean
        
        if std is not None:
            image = image / std
        
        image = np.moveaxis(image, -1, 0)
        

        labels = self.label_dict[name][max(0,i-step):min(i+step+1, n-1), :]
        
        if labels.ndim == 2:
            labels = labels.any(0).astype(np.int32)
            
            
        labels = torch.from_numpy(labels) 
        image = torch.from_numpy(image)
        
        return image, labels
    
    def __len__(self):

        return len(self.names)
    
