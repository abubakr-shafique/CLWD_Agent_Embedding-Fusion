import os, sys
import argparse
import numpy as np
import pandas as pd
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.utils.class_weight import compute_class_weight

import project_dirs as pdir
import config.MIL_config as MIL_config
import utils.fusion_data_utils as fusion_utils

logg = True

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser(description='CLWD Slide Embedding Extraction')
parser.add_argument('--embedding_dir', type=str, default=pdir.EMBEDDINGS_DIR)
parser.add_argument('--train_csv', type=str, default=MIL_config.train_csv)
parser.add_argument('--val_csv', type=str, default=MIL_config.val_csv)
parser.add_argument('--test_csv', type=str, default=MIL_config.test_csv)
parser.add_argument('--output_dir', type=str, default=pdir.ARTIFACTS_DIR)
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--seed', type=int, default=8, help="Random seed for reproducible experiment (default: 8)")
parser.add_argument('--target_mag', type=int, default=MIL_config.target_magnification)
parser.add_argument('--target_patch_size', type=int, default=MIL_config.target_patch_size)

args = parser.parse_args()

if __name__ == '__main__':
    print("Prepare Combined Data for Embedding Fusion.\n")
    set_seed(args.seed) # Set seed for reproducibil
    print(f"Fold: {args.fold}")
    FMs = ["UNI2-h", "Virchow2", "H-OPTIMUS-1"]

    train_csv_Data = pd.read_csv(os.path.join(pdir.CSV_DIR, f"Fold_{args.fold}", args.train_csv))
    val_csv_Data = pd.read_csv(os.path.join(pdir.CSV_DIR, f"Fold_{args.fold}", args.val_csv))
    test_csv_Data = pd.read_csv(os.path.join(pdir.CSV_DIR, f"Fold_{args.fold}", args.test_csv))

    train_WSIs = train_csv_Data["WSI_ID"].tolist()
    train_age = train_csv_Data["Age"].tolist()
    train_sex = train_csv_Data["Sex"].tolist()
    train_Labels = train_csv_Data["Benchmark_Label_7class"].tolist()

    val_WSIs = val_csv_Data["WSI_ID"].tolist()
    val_Labels = val_csv_Data["Benchmark_Label_7class"].tolist()

    test_WSIs = test_csv_Data["WSI_ID"].tolist()
    test_Labels = test_csv_Data["Benchmark_Label_7class"].tolist()

    label_names = sorted(set(train_Labels))
    n_classes = len(label_names)
    label_to_index = {label: idx for idx, label in enumerate(label_names)}
    index_to_label = {idx: label for idx, label in enumerate(label_names)}

    sex_names = sorted(set(train_sex))
    sex_to_index = {label: idx for idx, label in enumerate(sex_names)}
    index_to_sex = {idx: label for idx, label in enumerate(sex_names)}

    embedding_root_dir_UNI2 = os.path.join(args.embedding_dir, "slide", f"{FMs[0]}_{args.target_patch_size}_{args.target_mag}x", f"Fold_{args.fold}")
    embedding_root_dir_Virchow2 = os.path.join(args.embedding_dir, "slide", f"{FMs[1]}_{args.target_patch_size}_{args.target_mag}x", f"Fold_{args.fold}")
    embedding_root_dir_OPTIMUS = os.path.join(args.embedding_dir, "slide", f"{FMs[2]}_{args.target_patch_size}_{args.target_mag}x", f"Fold_{args.fold}")
    embedding_root_dir_MetaData = os.path.join(args.embedding_dir, "metadata_SexAge", f"AgeSex_Linear", f"Fold_{args.fold}")

    train_dataset = fusion_utils.Slide_Clinical_Dataset(train_csv_Data, label_to_index, sex_to_index,
                                                        embedding_root_FM1=embedding_root_dir_UNI2,
                                                        embedding_root_FM2=embedding_root_dir_Virchow2,
                                                        embedding_root_FM3=embedding_root_dir_OPTIMUS,
                                                        embedding_root_metadata = embedding_root_dir_MetaData)
    val_dataset = fusion_utils.Slide_Clinical_Dataset(val_csv_Data, label_to_index, sex_to_index,
                                                        embedding_root_FM1=embedding_root_dir_UNI2,
                                                        embedding_root_FM2=embedding_root_dir_Virchow2,
                                                        embedding_root_FM3=embedding_root_dir_OPTIMUS,
                                                        embedding_root_metadata = embedding_root_dir_MetaData)
    test_dataset = fusion_utils.Slide_Clinical_Dataset(test_csv_Data, label_to_index, sex_to_index,
                                                        embedding_root_FM1=embedding_root_dir_UNI2,
                                                        embedding_root_FM2=embedding_root_dir_Virchow2,
                                                        embedding_root_FM3=embedding_root_dir_OPTIMUS,
                                                        embedding_root_metadata = embedding_root_dir_MetaData)

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=False, drop_last=False)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, drop_last=False)

    fusion_data_save_dir = os.path.join(args.output_dir, "Fusion_Data", f"Fold_{args.fold}")
    os.makedirs(fusion_data_save_dir, exist_ok=True)

    new_data_dict = fusion_utils.data_to_dict(train_dataloader)
    torch.save(new_data_dict, os.path.join(fusion_data_save_dir, f"train.pt"))

    if logg:
        print("Train:")
        print("UNI2-h: ")
        print(new_data_dict["uni2"].shape)
        print("Virchow2: ")
        print(new_data_dict["virchow2"].shape)
        print("H-OPTIMUS-1: ")
        print(new_data_dict["optimus"].shape)
        print("[Age, Sex]: ")
        print(new_data_dict["meta"].shape)
        print("Labels: ")
        print(new_data_dict["labels"].shape)
        # print("WSI IDs: ")
        # print(new_data_dict["ids"])

    new_data_dict = fusion_utils.data_to_dict(val_dataloader)
    torch.save(new_data_dict, os.path.join(fusion_data_save_dir, f"val.pt"))

    if logg:
        print("Validation:")
        print("UNI2-h: ")
        print(new_data_dict["uni2"].shape)
        print("Virchow2: ")
        print(new_data_dict["virchow2"].shape)
        print("H-OPTIMUS-1: ")
        print(new_data_dict["optimus"].shape)
        print("[Age, Sex]: ")
        print(new_data_dict["meta"].shape)
        print("Labels: ")
        print(new_data_dict["labels"].shape)
        # print("WSI IDs: ")
        # print(new_data_dict["ids"])

    new_data_dict = fusion_utils.data_to_dict(test_dataloader)
    torch.save(new_data_dict, os.path.join(fusion_data_save_dir, f"test.pt"))

    if logg:
        print("Test:")
        print("UNI2-h: ")
        print(new_data_dict["uni2"].shape)
        print("Virchow2: ")
        print(new_data_dict["virchow2"].shape)
        print("H-OPTIMUS-1: ")
        print(new_data_dict["optimus"].shape)
        print("[Age, Sex]: ")
        print(new_data_dict["meta"].shape)
        print("Labels: ")
        print(new_data_dict["labels"].shape)
        # print("WSI IDs: ")
        # print(new_data_dict["ids"])