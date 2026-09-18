import os, sys
import argparse
import numpy as np
import pandas as pd
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from sklearn.utils.class_weight import compute_class_weight

import torchvision

import project_dirs as pdir
import config.slide_metadata_config as slide_metadata_config
import utils.MIL_utils as MIL_utils
import MIL_model.abmil as get_MIL
import utils.metadata_utils as metadata_utils
import utils.slide_metadata_utils as slide_metadata_utils

logg = False

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser(description='CLWD MIL Slide With Metadata Classifier.')
parser.add_argument('--embedding_dir', type=str, default=pdir.EMBEDDINGS_DIR)
parser.add_argument('--train_csv', type=str, default=slide_metadata_config.train_csv)
parser.add_argument('--val_csv', type=str, default=slide_metadata_config.val_csv)
parser.add_argument('--test_csv', type=str, default=slide_metadata_config.test_csv)
parser.add_argument('--output_dir', type=str, default=pdir.EVAL_DIR)
parser.add_argument('--result_dir', type=str, default=pdir.RESULT_DIR)
parser.add_argument('--model_name', type=str, default="H-OPTIMUS-1", choices=['H-OPTIMUS-1', 'UNI2-h', 'Virchow2'])
parser.add_argument('--mode', type=str, default="train", choices=['train', 'eval'])
parser.add_argument('--MIL_model', type=str, default=slide_metadata_config.MIL_Model, choices=['ABMIL'])
parser.add_argument('--metadata_model', type=str, default='AgeSex_Linear', choices=['AgeSex_Linear'])
parser.add_argument('--pretrained', type=str, default="yes", choices=['yes', 'no'])
parser.add_argument('--freeze_models', type=str, default="yes", choices=['yes', 'no'])
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--seed', type=int, default=8, help="Random seed for reproducible experiment (default: 8)")
parser.add_argument('--target_mag', type=int, default=slide_metadata_config.target_magnification)
parser.add_argument('--target_patch_size', type=int, default=slide_metadata_config.target_patch_size)

args = parser.parse_args()

if __name__ == '__main__':
    print("CLWD MIL Slide With Metadata Classifier.\n")
    set_seed(args.seed) # Set seed for reproducibility

    print(f"Embedding Model: {args.model_name}, MIL Model: {args.MIL_model}, MetaData Model: {args.metadata_model}, Fold: {args.fold}")

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


    embedding_root_folder = os.path.join(args.embedding_dir, "slide", f"{args.model_name}_{args.target_patch_size}_{args.target_mag}x", f"Fold_{args.fold}")
    metadata_embedding_root_folder = os.path.join(args.embedding_dir, "metadata_SexAge", f"{args.metadata_model}", f"Fold_{args.fold}")

    train_dataset = slide_metadata_utils.Slide_Clinical_Dataset_Embeddings(train_csv_Data, label_to_index, sex_to_index, embedding_root_slide=embedding_root_folder, embedding_root_metadata=metadata_embedding_root_folder)
    val_dataset = slide_metadata_utils.Slide_Clinical_Dataset_Embeddings(val_csv_Data, label_to_index, sex_to_index, embedding_root_slide=embedding_root_folder, embedding_root_metadata=metadata_embedding_root_folder)
    test_dataset = slide_metadata_utils.Slide_Clinical_Dataset_Embeddings(test_csv_Data, label_to_index, sex_to_index, embedding_root_slide=embedding_root_folder, embedding_root_metadata=metadata_embedding_root_folder)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # Calculate weights ONLY from the training data. 
    class_weights = compute_class_weight(class_weight="balanced", classes=np.arange(n_classes), y=train_dataset.y) 
    class_weights = torch.tensor( class_weights, dtype=torch.float32 )
    if logg:
        print(f"Class weights: {class_weights}")

    

    Slide_MetaData_Combined_model = slide_metadata_utils.Slide_MetaData_Classifier_Embeddings()

    if logg:
            print(f"Slide MetaData Combined Model\n")
            print(Slide_MetaData_Combined_model)
            print("\n\n")

    checkpoint_save_dir = os.path.join(args.output_dir, "Slide_MetaData_checkpoints", f"{args.model_name}_{slide_metadata_config.MIL_Model}_{args.metadata_model}")
    os.makedirs(checkpoint_save_dir, exist_ok=True)

    if args.mode == "train":
        print(f"Training {slide_metadata_config.MIL_Model} and {args.metadata_model} combined")
        ##train the MIL model
        slide_metadata_utils.train_loop(model=Slide_MetaData_Combined_model, data_loader_Train=train_dataloader,
                                        model_name=f"{args.MIL_model}_{args.metadata_model}",
                                        data_loader_Val=val_dataloader, cls_weights=class_weights, input_dim=1024,
                                        output_dir=checkpoint_save_dir, Fold=f"Fold_{args.fold}")

    else:
        print(f"Evaluating {slide_metadata_config.MIL_Model} and {args.metadata_model} combined")
        ### Evaluate
        results_save_dir = os.path.join(args.result_dir, "Slide_MetaData_results", f"{args.model_name}_{slide_metadata_config.MIL_Model}_{args.metadata_model}")
        os.makedirs(results_save_dir, exist_ok=True)

        try:
            # Load the best state dictionary from a file
            checkpoint_load_dir = os.path.join(checkpoint_save_dir, f"{args.MIL_model}_{args.metadata_model}", f"Fold_{args.fold}")
            state_dict = torch.load(os.path.join(checkpoint_load_dir, f"{args.MIL_model}_{args.metadata_model}_Classifier.pth"))
            # Load the state dictionary into the model
            Slide_MetaData_Combined_model.load_state_dict(state_dict, strict=True)

            eval_history = slide_metadata_utils.eval_loop(model=Slide_MetaData_Combined_model, data_loader_Test=test_dataloader, Class_idx_to_Label=index_to_label, model_name=f"{args.MIL_model}_{args.metadata_model}",
                                            Class_Label_to_idx=label_to_index, results_dir=results_save_dir, n_classes=n_classes, label_names=label_names, Fold=f"Fold_{args.fold}")

        except:
            print("No trained MIL model found / or / Evaluation Error.")
