# 8th Place Solution - RSNA Intracranial Aneurysm Detection
[[Competition]](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection)

Training Code for 8th Place Solution - Kaggle Challenge: **RSNA Intracranial Aneurysm Detection**

## 📁 Project Structure

```
repo/
├── content
├── model
├── src
└── data
    ├─── series/
    ├─── train.csv
    └─── train_localizers.csv

```

Download on [Kaggle](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/data) the `rsna-intracranial-aneurysm-detection.zip` data and extract `series`, `train.csv` and `train_localizers.csv` to `data`folder.


## 📜 Skripts Order

1. `preprocess_data.py`
2. `create_folds.py` 
3. `train_stage1.py` and `eval_stage1.py`
4. `extract_features.py`
5. `train_stage2.py` and `eval_stage2.py`
6. `eval_end_to_end.py` 


### 1. [preprocess_data.py](preprocess_data.py): 
Load all dicom series and saves the resized `n x 512 x 512` series to `./data/npy_sequence/*.npy` and every single image to `./data/npy_slice/{UID}/*.npy` as numpy float16 array for faster dataloading during training.

### 2. [create_folds.py](create_folds.py): 
Define folds and create `./data/train_stage1_{FOLDS}_folds.csv"` for training of `stage1` and defines spots in each sequence where negative cases (no aneurysm present) can be sampled.  

### 3. [train_stage1.py](train_stage1.py):
Train a classifier by sampling positve spots (given by `train_localizers.csv`) and negative spots (output of `create_folds.py`). 

<img src="content/stage1.png" alt="drawing" width="800"/>

Use the provided file `./data/f_dict_hard.pkl` during training of `stage1` to sample difficult regions where no aneurysm is present but based on OOF predictions of an early 5 fold classifier the probability for an aneurysm was above a certain threshold. 

<img src="content/sampling.png" alt="drawing" width="800"/>

### 3. [eval_stage1.py](eval_stage1.py):
Evalute model after `stage1` training. Scores are much higher than after `stage2` because we only evaluate here on the given spots, what means for positive cases we hit right the spot where an aneurysm is present.

### 4. [extract_features.py](extract_features.py):
Extracts the features before the classification head as intput of `stage2` transformer. 

### 5. [train_stage2.py](train_stage2.py):
Train a classifier on the whole sequence per uid based on the extracted features of `stage1`.

<img src="content/stage2.png" alt="drawing" width="800"/>

### 5. [eval_stage2.py](eval_stage2.py):
Evalute model after `stage2` training on the pre-extraced features of `stage1`. Scores now reflect the final scores like during inference of unseen data, where we do not know where or if in the whole sequence an aneurysm is present.


### 6. [eval_end_to_end.py](eval_end_to_end.py):
Evaluate the `stage1` + `stage2` by loading dicom series and do ent-to-end evaluation. 