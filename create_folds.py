import pandas as pd
from glob import glob
import os
import numpy as np
import pickle 
import pandas as pd
from collections import defaultdict
from glob import glob
import numpy as np
import pickle 
import random

FOLDS = 10

#---------------------------------------------------------------------------------------------------#
# Load data                                                                                         #
#---------------------------------------------------------------------------------------------------#

df_train = pd.read_csv("./data/train_meta.csv")
df_loc = pd.read_csv("./data/train_localizers.csv")

# Slice uids created in preprocess_data.py
with open("./data/slice_uids_dict.pkl", "rb") as f:
    series_dict = pickle.load(f)
   

#---------------------------------------------------------------------------------------------------#
# Set seed                                                                                          #
#---------------------------------------------------------------------------------------------------#   

random.seed(42)
np.random.seed(42)


#---------------------------------------------------------------------------------------------------#
# Class to class_id                                                                                 #
#---------------------------------------------------------------------------------------------------#

classes = [
           'Left Infraclinoid Internal Carotid Artery',
           'Right Infraclinoid Internal Carotid Artery',
           'Left Supraclinoid Internal Carotid Artery',
           'Right Supraclinoid Internal Carotid Artery',
           'Left Middle Cerebral Artery',
           'Right Middle Cerebral Artery',
           'Anterior Communicating Artery',
           'Left Anterior Cerebral Artery',
           'Right Anterior Cerebral Artery',
           'Left Posterior Communicating Artery',
           'Right Posterior Communicating Artery',
           'Basilar Tip',
           'Other Posterior Circulation',
           'Aneurysm Present',
]

classes2id = dict(zip(classes, np.arange(len(classes))))

#---------------------------------------------------------------------------------------------------#
# Create folds                                                                                      #
#---------------------------------------------------------------------------------------------------#
 
sort_list = ['Aneurysm Present',
             'Modality',
             'PatientSex',
             'Anterior Communicating Artery',
             'Basilar Tip',
             'Other Posterior Circulation',
             'Left Infraclinoid Internal Carotid Artery',
             'Right Infraclinoid Internal Carotid Artery',
             'Left Supraclinoid Internal Carotid Artery',
             'Right Supraclinoid Internal Carotid Artery',
             'Left Middle Cerebral Artery',
             'Right Middle Cerebral Artery',
             'Left Anterior Cerebral Artery',
             'Right Anterior Cerebral Artery',
             'Left Posterior Communicating Artery',
             'Right Posterior Communicating Artery',
             ]

df_sort = df_train.sort_values(sort_list)

df_train["fold"] = -1

for i in range(FOLDS):
    
    select = df_sort.index[i::FOLDS]
    
    df_train.loc[select,"fold"] = i
    
        
df_train.to_csv(f"./data/train_{FOLDS}_folds.csv", index=False)

#---------------------------------------------------------------------------------------------------#
# Create Dataframe for Stage-I                                                                      #
#---------------------------------------------------------------------------------------------------#

uid2fold   = dict(zip(df_train['SeriesInstanceUID'], df_train["fold"]))
uid2length = dict(zip(df_train['SeriesInstanceUID'], df_train["length"]))
uid2height = dict(zip(df_train['SeriesInstanceUID'], df_train['h'] ))
uid2width  = dict(zip(df_train['SeriesInstanceUID'], df_train['w'] )) 
uid2mod    = dict(zip(df_train['SeriesInstanceUID'], df_train["mod"]))


df_loc["fold"] = df_loc['SeriesInstanceUID'].map(uid2fold)
df_loc["length"] = df_loc['SeriesInstanceUID'].map(uid2length)

x_list = []
y_list = []
f_list = []


for i in range(len(df_loc)):
    
    row = df_loc.iloc[i]
    coordinates = eval(row['coordinates'])

    x_list.append(coordinates["x"])
    y_list.append(coordinates["y"])
    f_list.append(coordinates.get("f", -1))
    
    
df_loc["x"] = x_list
df_loc["y"] = y_list
df_loc["f"] = f_list  


#---------------------------------------------------------------------------------------------------#
# Series with aneurysm positiv spot                                                                 #
#---------------------------------------------------------------------------------------------------#
  
f_list = []
    
for i in range(len(df_loc)):
    
    row = df_loc.iloc[i]
    series = row['SeriesInstanceUID']
    instance_uid = row['SOPInstanceUID']
    seq_uids = series_dict.get(series, None)
     
    f = row['f']
    
    if f == -1:
        match = (np.array(seq_uids) == instance_uid).astype(np.int32)
        f = match.argmax()
        f_list.append(int(f))
        
    else:
        
        f_list.append(f)
   
        
df_loc["f"] = f_list
df_loc["f_rel"] = df_loc["f"] / df_loc["length"]
df_loc["class_id"] = df_loc["location"].map(classes2id)


#---------------------------------------------------------------------------------------------------#
# Series without aneurysm negativ spot
#---------------------------------------------------------------------------------------------------#

df_train[df_train['Aneurysm Present']==0]['SeriesInstanceUID'].values

df_loc_neg = pd.DataFrame({'SeriesInstanceUID': df_train[df_train['Aneurysm Present']==0]['SeriesInstanceUID'].values,
                           'SOPInstanceUID': '',
                           'coordinates': '',
                           'location': 'Nothing',
                           'fold': 0,
                           'x': 0,
                           'y': 0, 
                           'f': -2,
                           'length': -1,
                           "f_rel": -2.0,
                           'class_id': 13
                           })

df_loc_neg["fold"] = df_loc_neg['SeriesInstanceUID'].map(uid2fold)
df_loc_neg["length"] = df_loc_neg['SeriesInstanceUID'].map(uid2length)



#---------------------------------------------------------------------------------------------------#
# Series with aneurysm negativ spot                                                                 #
#---------------------------------------------------------------------------------------------------#

df_loc_neg_pos = df_loc.copy()

df_loc_neg_pos['SOPInstanceUID'] = ''
df_loc_neg_pos['coordinates'] = ''
df_loc_neg_pos['location'] = 'Nothing'
df_loc_neg_pos['x'] = 0
df_loc_neg_pos['y'] = 0
df_loc_neg_pos['f'] = -1
df_loc_neg_pos["f_rel"] = -1.0
df_loc_neg_pos['class_id'] = 13

#---------------------------------------------------------------------------------------------------#
# Combine dataframe for Stage-I                                                                     #
#---------------------------------------------------------------------------------------------------#

df_loc_train = pd.concat([df_loc, df_loc_neg, df_loc_neg_pos]).reset_index(drop=True)


df_loc_train["h"] = df_loc_train['SeriesInstanceUID'].map(uid2height)
df_loc_train["w"] = df_loc_train['SeriesInstanceUID'].map(uid2width)



df_loc_train["x_512"] = (df_loc_train["x"] / df_loc_train["w"] * 512).round().astype(np.int32)
df_loc_train["y_512"] = (df_loc_train["y"] / df_loc_train["h"] * 512).round().astype(np.int32)

df_loc_train["mod"] = df_loc_train['SeriesInstanceUID'].map(uid2mod)



# Check left <-> right normalized to 512x512 input size

df_loc_pos = df_loc_train[df_loc_train["class_id"]!=13]

stats = list()

for loc in df_loc_pos['location'].unique():
    
    df_tmp = df_loc_pos[df_loc_pos['location']==loc]
    
    stats.append({'location': loc,
                  "x": df_tmp["x_512"].mean(),
                  "y": df_tmp["y_512"].mean()})
    
    
df_side = pd.DataFrame(stats)
    
df_side.to_csv("./data/label_side.csv", index=False)


#---------------------------------------------------------------------------------------------------#
# Calculate spots per sequence where we can savely sample negative cases from                       #
# also set per sequences where we have no aneurysm a spot to use for evaluation of Stage-I          #
#---------------------------------------------------------------------------------------------------#

f_dict = defaultdict(list)
f_class_dict = defaultdict(dict)


for i in range(len(df_loc_pos)):
    
    row = df_loc_pos.iloc[i]
    series = row["SeriesInstanceUID"]
    f_dict[series].append(int(row["f"]))
    
    try:
        f_class_dict[series][int(row["f"])].append(int(row["class_id"]))
    except:
        f_class_dict[series][int(row["f"])] = [int(row["class_id"])]
    

f_rel = df_loc_pos["f_rel"].values.tolist()
f_rel = f_rel * 3   
    

df_keep_list = []

sample_dict = dict()
label_dict = dict()


for i in range(len(df_loc_train)):

    row = df_loc_train.iloc[i] 
    length = row["length"]
    series = row["SeriesInstanceUID"] 
    sample_dict[series] = np.arange(length)
    label_dict[series] = np.zeros((length, 14), dtype=np.uint8)
    
    
x_list = []  

for i in range(len(df_loc_train)):
    
    row = df_loc_train.iloc[i]
    series = row["SeriesInstanceUID"]
    marker = f_dict[series]
    n = row["length"]
    length = row["length"]
    class_id = row["class_id"]
    f = row["f"]
    
    if class_id != 13:
        label_dict[series][f, [class_id, 13]] = 1
        x_list.append(label_dict[series])
        
    m = max(5, int(length * 0.05))
    
    for x in marker:
        start = max(0, x - m)
        end = min(x + m + 1, n-1)
        sample_dict[series][start:end] = -1
        

sample_dict_keep = dict()  
delete_list = list()   
        
for i in range(len(df_loc_train)):
    
    row = df_loc_train.iloc[i].copy()
    series = row["SeriesInstanceUID"]
    length = row["length"]   
    spots = sample_dict[series].copy()
          
    select = spots != -1
    delete = spots == -1 
    delete_list.append(int(delete.sum()))
    spots_select = spots[select]
    
    if row["f"] == -2:
        row["f"] = int(f_rel[i] * length)
           
    if len(spots_select) > 5:
    
        sample_dict_keep[series] = spots_select
        
        df_keep_list.append(dict(row))
        
#---------------------------------------------------------------------------------------------------#
# Save DataFrame and also dictionaries for sampling negativ spots                                   #
#---------------------------------------------------------------------------------------------------#
   
with open("./data/label_dict.pkl", "wb") as f:
    pickle.dump(label_dict, f)   

with open("./data/f_dict_sample.pkl", "wb") as f:
    pickle.dump(sample_dict_keep, f) 

with open("./data/f_dict.pkl", "wb") as f:
    pickle.dump(f_dict, f)     
             
df_cls_train = pd.DataFrame(df_keep_list)
df_cls_train.to_csv(f"./data/train_stage1_{FOLDS}_folds.csv", index=False)







