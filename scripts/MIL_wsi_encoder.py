import os, sys
import argparse
import numpy as np
import pandas as pd
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchinfo import summary

import torchvision

import project_dirs as pdir
import config.MIL_config as MIL_config
import utils.MIL_utils as MIL_utils
import MIL_model.abmil as get_MIL

logg = False

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser(description='CLWD Slide Embedding Extraction')
parser.add_argument('--embedding_dir', type=str, default=MIL_config.embedding_dir)
parser.add_argument('--train_csv', type=str, default=MIL_config.train_csv)
parser.add_argument('--val_csv', type=str, default=MIL_config.val_csv)
parser.add_argument('--test_csv', type=str, default=MIL_config.test_csv)
parser.add_argument('--output_dir', type=str, default=MIL_config.output_dir)
parser.add_argument('--result_dir', type=str, default=MIL_config.result_dir)
parser.add_argument('--model_name', type=str, default='UNI2-h', choices=['H-OPTIMUS-1', 'UNI2-h', 'Virchow2'])
parser.add_argument('--mode', type=str, default="train", choices=['train', 'eval'])
parser.add_argument('--MIL_model', type=str, default=MIL_config.MIL_Model, choices=['ABMIL'])
parser.add_argument('--batch_size', type=int, default=MIL_config.batch_size)
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--seed', type=int, default=8, help="Random seed for reproducible experiment (default: 8)")
parser.add_argument('--target_mag', type=int, default=MIL_config.target_magnification)
parser.add_argument('--target_patch_size', type=int, default=MIL_config.target_patch_size)

args = parser.parse_args()

if __name__ == '__main__':
    print("CLWD MIL Slide Embeddings.\n")
    set_seed(args.seed) # Set seed for reproducibility

    train_csv_Data = pd.read_csv(os.path.join(pdir.CSV_DIR, f"Fold_{args.fold}", args.train_csv))
    val_csv_Data = pd.read_csv(os.path.join(pdir.CSV_DIR, f"Fold_{args.fold}", args.val_csv))
    test_csv_Data = pd.read_csv(os.path.join(pdir.CSV_DIR, f"Fold_{args.fold}", args.test_csv))

    train_WSIs = train_csv_Data["WSI_ID"].tolist()
    train_Labels = train_csv_Data["Benchmark_Label_7class"].tolist()

    val_WSIs = val_csv_Data["WSI_ID"].tolist()
    val_Labels = val_csv_Data["Benchmark_Label_7class"].tolist()

    test_WSIs = test_csv_Data["WSI_ID"].tolist()
    test_Labels = test_csv_Data["Benchmark_Label_7class"].tolist()

    label_names = sorted(set(train_Labels))
    n_classes = len(label_names)
    label_to_index = {label: idx for idx, label in enumerate(label_names)}
    index_to_label = {idx: label for idx, label in enumerate(label_names)}

    train_Labels_idx = MIL_utils.labels_to_indices(train_Labels, label_to_index)
    val_Labels_idx = MIL_utils.labels_to_indices(val_Labels, label_to_index)
    test_Labels_idx = MIL_utils.labels_to_indices(test_Labels, label_to_index)

    if logg:
        print(f"Number of Classes: {n_classes}")
        print(index_to_label)

    embedding_root_folder = os.path.join(args.embedding_dir, "patch", f"{args.model_name}_{args.target_patch_size}_{args.target_mag}x")
    train_embedding_paths = MIL_utils.make_path_list(embedding_root_folder, train_WSIs, file_ext=".pt")
    val_embedding_paths = MIL_utils.make_path_list(embedding_root_folder, val_WSIs, file_ext=".pt")
    test_embedding_paths = MIL_utils.make_path_list(embedding_root_folder, test_WSIs, file_ext=".pt")

    coords_root_folder = pdir.PATCH_COORDS_DIR
    train_coords_paths = MIL_utils.make_path_list(coords_root_folder, train_WSIs, file_ext=".npy")
    val_coords_paths = MIL_utils.make_path_list(coords_root_folder, val_WSIs, file_ext=".npy")
    test_coords_paths = MIL_utils.make_path_list(coords_root_folder, test_WSIs, file_ext=".npy")

    # Create dataset
    train_dataset = MIL_utils.EmbeddingDataset(embedding_paths=train_embedding_paths, coord_paths=train_coords_paths, labels=train_Labels_idx)
    val_dataset = MIL_utils.EmbeddingDataset(embedding_paths=val_embedding_paths, coord_paths=val_coords_paths, labels=val_Labels_idx)
    test_dataset = MIL_utils.EmbeddingDataset(embedding_paths=test_embedding_paths, coord_paths=test_coords_paths, labels=test_Labels_idx)

    class_weights = MIL_utils.compute_class_weights(train_dataset)
    if logg:
        print(f"class Weights: {class_weights}")
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

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
    if logg:
        print(MIL_model)

    checkpoint_save_dir = os.path.join(args.output_dir, "MIL_checkpoints", f"{args.model_name}_{MIL_config.MIL_Model}_{args.seed}")
    os.makedirs(checkpoint_save_dir, exist_ok=True)

    if args.mode == "train":
        print(f"Training {MIL_config.MIL_Model}")
        ##train the MIL model
        MIL_utils.train_loop(model=MIL_model, data_loader_Train=train_dataloader, model_name=args.model_name,
                            data_loader_Val=val_dataloader, cls_weights=class_weights, input_dim=input_dim,
                            output_dir=checkpoint_save_dir, Fold=f"Fold_{args.fold}")
    else:
        print(f"Evaluating {MIL_config.MIL_Model}")
        ### Evaluate
        results_save_dir = os.path.join(args.result_dir, "MIL_results", f"{args.model_name}_ABMIL_{args.seed}")
        os.makedirs(results_save_dir, exist_ok=True)

        try:
            # Load the best state dictionary from a file
            checkpoint_load_dir = os.path.join(checkpoint_save_dir, args.model_name, f"Fold_{args.fold}")
            state_dict = torch.load(os.path.join(checkpoint_load_dir, f"{args.model_name}_Classifier.pth"))
            # Load the state dictionary into the model
            MIL_model.load_state_dict(state_dict, strict=True)
            MIL_model = MIL_model.to(device)

            eval_history = MIL_utils.eval_loop(model=MIL_model, data_loader_Test=test_dataloader, Class_idx_to_Label=index_to_label, model_name=args.model_name,
                                               Class_Label_to_idx=label_to_index, results_dir=results_save_dir, n_classes=n_classes, label_names=label_names, Fold=f"Fold_{args.fold}")

        except:
            print("No trained MIL model found / or / Evaluation Error.")
