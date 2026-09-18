import os, sys
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import f1_score, balanced_accuracy_score

from pathlib import Path

# Get the absolute path of the parent directory
parent_dir = Path(__file__).resolve().parent.parent
# Add the parent directory to sys.path if it's not already there
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

class Slide_Clinical_Dataset(Dataset):

    def __init__(self, df, label_to_index, sex_to_index, embedding_root_FM1, embedding_root_FM2, embedding_root_FM3, embedding_root_metadata, file_ext=".pt", device='cpu'):

        self.df = df.reset_index(drop=True)
        self.embedding_root_FM1 = embedding_root_FM1
        self.embedding_root_FM2 = embedding_root_FM2
        self.embedding_root_FM3 = embedding_root_FM3
        self.embedding_root_metadata = embedding_root_metadata
        self.file_ext = file_ext
        self.device = device

        # ----------------------------------------------------
        # Age normalization
        # Age is explicitly normalized from 0-100 to 0-1
        # ----------------------------------------------------
        age = self.df["Age"].astype(float).values
        age = np.clip(age, 0, 100)
        age = age / 100.0

        # ----------------------------------------------------
        # Sex encoding
        # Female = 0
        # Male   = 1
        # ----------------------------------------------------
        sex = self.df["Sex"].map(sex_to_index).values

        self.WSI_ID = self.df["WSI_ID"].astype(str).values

        # Features: [normalized_age, sex]
        self.X = np.column_stack([
            age,
            sex
        ]).astype(np.float32)

        self.y = (
            self.df["Benchmark_Label_7class"]
            .map(label_to_index)
            .values
            .astype(np.int64)
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        FM1_embedding_path = os.path.join(self.embedding_root_FM1, f"{self.WSI_ID[idx]}{self.file_ext}")
        FM1_embedding = torch.load(FM1_embedding_path, map_location=self.device) ## for pt file

        FM2_embedding_path = os.path.join(self.embedding_root_FM2, f"{self.WSI_ID[idx]}{self.file_ext}")
        FM2_embedding = torch.load(FM2_embedding_path, map_location=self.device) ## for pt file

        FM3_embedding_path = os.path.join(self.embedding_root_FM3, f"{self.WSI_ID[idx]}{self.file_ext}")
        FM3_embedding = torch.load(FM3_embedding_path, map_location=self.device) ## for pt file

        # Age_sex = torch.tensor(self.X[idx], dtype=torch.float32)
        metadata_embedding_path = os.path.join(self.embedding_root_metadata, f"{self.WSI_ID[idx]}{self.file_ext}")
        Age_sex = torch.load(metadata_embedding_path, map_location=self.device) ## for pt file

        label = torch.tensor(self.y[idx], dtype=torch.long)

        return self.WSI_ID[idx], FM1_embedding, FM2_embedding, FM3_embedding, Age_sex, label


def data_to_dict(dataloader):

    # Store data as lists during the loop
    new_data_dict = {
        "uni2": [],
        "virchow2": [],
        "optimus": [],
        "meta": [],
        "labels": [],
        "ids": []
    }

    data_loop = tqdm(dataloader)

    for idx, (WSI_ID, FM1_embedding, FM2_embedding,
              FM3_embedding, Age_sex, label) in enumerate(data_loop):

        for i, WSI in enumerate(WSI_ID):

            new_data_dict["ids"].append(WSI_ID[i])
            
            new_data_dict["uni2"].append(FM1_embedding[i].squeeze())
            new_data_dict["virchow2"].append(FM2_embedding[i].squeeze())
            new_data_dict["optimus"].append(FM3_embedding[i].squeeze())
            new_data_dict["meta"].append(Age_sex[i])
            new_data_dict["labels"].append(label[i])

    # Convert lists of tensors into tensors
    new_data_dict["uni2"] = torch.stack(new_data_dict["uni2"])
    new_data_dict["virchow2"] = torch.stack(new_data_dict["virchow2"])
    new_data_dict["optimus"] = torch.stack(new_data_dict["optimus"])
    new_data_dict["meta"] = torch.stack(new_data_dict["meta"])
    new_data_dict["labels"] = torch.stack(new_data_dict["labels"])

    return new_data_dict