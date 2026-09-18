import os, sys
import argparse
import time
import pandas as pd
from tqdm import tqdm
import numpy as np
import random

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


import project_dirs as pdir
import config.patch_embed_gen_config as patch_config
import config.MIL_config as MIL_config
import utils.MIL_utils as MIL_utils
import MIL_model.abmil as get_MIL
import config.metadata_classifier_config as Meta_config
# import FMs.load_models as LM
import utils.metadata_utils as metadata_utils

logg = False

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser(description='CLWD Patch Embedding Extraction')
parser.add_argument('--embedding_dir', type=str, default=pdir.EMBEDDINGS_DIR)
parser.add_argument('--checkpoint_dir', type=str, default=MIL_config.output_dir)
parser.add_argument('--csv_dir', type=str, default=os.path.join(pdir.CSV_DIR, "CLWD_OneSlide.csv"))
parser.add_argument('--model_name', type=str, default='UNI2-h', choices=['H-OPTIMUS-1', 'UNI2-h', 'Virchow2', 'AgeSex_Linear'])
parser.add_argument('--mode', type=str, default='slide', choices=['slide', 'metadata'])
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--fold', type=int, default=1)
parser.add_argument('--seed', type=int, default=8, help="Random seed for reproducible experiment (default: 8)")
parser.add_argument('--target_patch_size', type=int, default=patch_config.target_patch_size)
parser.add_argument('--target_mag', type=int, default=patch_config.target_mag)

args = parser.parse_args()

if __name__ == '__main__':
    print("CLWD Generate Slide Embeddings.\n")
    set_seed(args.seed) # Set seed for reproducibility

    # Get WSI_names by using CSV File
    print(f"{args.model_name} Fold: {args.fold}")
    csv_data = pd.read_csv(args.csv_dir)
    WSI_names = csv_data["WSI_ID"].to_list()
    WSI_labels = csv_data["Benchmark_Label_7class"].to_list()

    label_names = sorted(set(WSI_labels))
    n_classes = len(label_names)
    label_to_index = {label: idx for idx, label in enumerate(label_names)}
    index_to_label = {idx: label for idx, label in enumerate(label_names)}

    WSI_labels_idx = MIL_utils.labels_to_indices(WSI_labels, label_to_index)

    if args.mode == "slide":
        embedding_root_folder = os.path.join(args.embedding_dir, "patch", f"{args.model_name}_{args.target_patch_size}_{args.target_mag}x")
        train_embedding_paths = MIL_utils.make_path_list(embedding_root_folder, WSI_names, file_ext=".pt")

        coords_root_folder = pdir.PATCH_COORDS_DIR
        train_coords_paths = MIL_utils.make_path_list(coords_root_folder, WSI_names, file_ext=".npy")

        WSI_dataset = MIL_utils.EmbeddingDataset(embedding_paths=train_embedding_paths, coord_paths=train_coords_paths, labels=WSI_labels_idx, return_path=True)

        WSI_dataloader = DataLoader(WSI_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

        if args.model_name == "Virchow2":
            input_dim = 1280
        elif args.model_name == "UNI2-h" or args.model_name == "H-OPTIMUS-1":
            input_dim = 1536
        else:
            input_dim = 1536

        MIL_model = get_MIL.ABMIL(in_dim=input_dim, num_classes=n_classes,
                                    embed_dim=MIL_config.embed_dim,
                                    num_fc_layers = MIL_config.num_fc_layers,
                                    dropout = MIL_config.dropout,
                                    attn_dim = MIL_config.attn_dim,
                                    gate = MIL_config.attn_dim)

        checkpoint_save_dir = os.path.join(args.checkpoint_dir, "MIL_checkpoints", f"{args.model_name}_{MIL_config.MIL_Model}_{args.seed}", args.model_name, f"Fold_{args.fold}", f"{args.model_name}_Classifier.pth")
        state_dict = torch.load(checkpoint_save_dir)
        # Load the state dictionary into the model
        MIL_model.load_state_dict(state_dict, strict=True)

        output_embed_dir = os.path.join(args.embedding_dir, "slide", f"{args.model_name}_{args.target_patch_size}_{args.target_mag}x", f"Fold_{args.fold}")
        os.makedirs(output_embed_dir, exist_ok=True)
        MIL_utils.generate_slide_embeddings(model=MIL_model, data_loader=WSI_dataloader, output_dir=output_embed_dir)


    elif args.mode == "metadata":
        # Get WSI_names by using CSV File
        WSI_sex = csv_data["Sex"].tolist()
        WSI_age = csv_data["Age"].tolist()

        sex_names = sorted(set(WSI_sex))
        sex_to_index = {label: idx for idx, label in enumerate(sex_names)}
        index_to_sex = {idx: label for idx, label in enumerate(sex_names)}

        metadata_dataset = metadata_utils.ClinicalDataset(csv_data, label_to_index, sex_to_index, has_labels=True, get_WSI_ID = True)

        metadata_dataloader = DataLoader(metadata_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

        model = metadata_utils.AgeSexClassifier(num_classes=n_classes)

        checkpoint_load_dir = os.path.join(args.checkpoint_dir, "Clinical_metadata_classifier", f"{args.model_name}", f"Fold_{args.fold}")
        state_dict = torch.load(os.path.join(checkpoint_load_dir, f"{args.model_name}_Representation.pth"))
        # Load the state dictionary into the model
        model.load_state_dict(state_dict, strict=True)

        output_embed_dir = os.path.join(args.embedding_dir, "metadata_SexAge", f"{args.model_name}", f"Fold_{args.fold}")
        os.makedirs(output_embed_dir, exist_ok=True)

        metadata_utils.generate_metadata_embeddings(model=model, metadata_loader=metadata_dataloader, output_dir=output_embed_dir)