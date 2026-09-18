import os, sys
import argparse
import numpy as np
import pandas as pd
import random

import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader 

from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report 
from sklearn.utils.class_weight import compute_class_weight

import project_dirs as pdir
import config.metadata_classifier_config as Meta_config
import utils.MIL_utils as MIL_utils
import utils.metadata_utils as metadata_utils

logg = True

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser(description='CLWD Slide Classification using Metadata (age and sex)')
parser.add_argument('--train_csv', type=str, default=Meta_config.train_csv)
parser.add_argument('--val_csv', type=str, default=Meta_config.val_csv)
parser.add_argument('--test_csv', type=str, default=Meta_config.test_csv)
parser.add_argument('--model_name', type=str, default=Meta_config.model_name, choices=['AgeSex_Linear'])
parser.add_argument('--output_dir', type=str, default=Meta_config.output_dir)
parser.add_argument('--result_dir', type=str, default=Meta_config.result_dir)
parser.add_argument('--mode', type=str, default="train", choices=['train', 'eval'])
parser.add_argument('--rep_learn', type=str, default="yes", choices=['yes', 'no'])
parser.add_argument('--batch_size', type=int, default=Meta_config.batch_size)
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--seed', type=int, default=8, help="Random seed for reproducible experiment (default: 8)")

args = parser.parse_args()

if __name__ == '__main__':
    print("CLWD Slide Classification using Metadata (age and sex).\n")
    set_seed(args.seed) # Set 

    train_csv_Data = pd.read_csv(os.path.join(pdir.CSV_DIR, f"Fold_{args.fold}", args.train_csv))
    val_csv_Data = pd.read_csv(os.path.join(pdir.CSV_DIR, f"Fold_{args.fold}", args.val_csv))
    test_csv_Data = pd.read_csv(os.path.join(pdir.CSV_DIR, f"Fold_{args.fold}", args.test_csv))

    train_WSIs = train_csv_Data["WSI_ID"].tolist()
    train_age = train_csv_Data["Age"].tolist()
    train_sex = train_csv_Data["Sex"].tolist()
    train_Labels = train_csv_Data["Benchmark_Label_7class"].tolist()

    label_names = sorted(set(train_Labels))
    NUM_CLASSES = len(label_names)
    label_to_index = {label: idx for idx, label in enumerate(label_names)}
    index_to_label = {idx: label for idx, label in enumerate(label_names)}

    sex_names = sorted(set(train_sex))
    sex_to_index = {label: idx for idx, label in enumerate(sex_names)}
    index_to_sex = {idx: label for idx, label in enumerate(sex_names)}

    if logg:
        print(f"Number of Classes: {NUM_CLASSES}")
        print(index_to_label)
        print(index_to_sex)

    
    train_dataset = metadata_utils.ClinicalDataset(train_csv_Data, label_to_index, sex_to_index, has_labels=True)
    val_dataset = metadata_utils.ClinicalDataset(val_csv_Data, label_to_index, sex_to_index, has_labels=True)
    test_dataset = metadata_utils.ClinicalDataset(test_csv_Data, label_to_index, sex_to_index, has_labels=True)

    train_dataloader = DataLoader(train_dataset, batch_size=Meta_config.batch_size, shuffle=True, drop_last=False)
    val_dataloader = DataLoader(val_dataset, batch_size=Meta_config.batch_size, shuffle=True, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=Meta_config.batch_size, shuffle=False, drop_last=False)

    # Calculate weights ONLY from the training data. 
    class_weights = compute_class_weight(class_weight="balanced", classes=np.arange(NUM_CLASSES), y=train_dataset.y) 
    class_weights = torch.tensor( class_weights, dtype=torch.float32 )
    if logg:
        print(f"Class weights: {class_weights}")

    model = metadata_utils.AgeSexClassifier(num_classes=NUM_CLASSES) ## Custom Linear CLassifier        

    if logg:
        print(model)

    checkpoint_save_dir = os.path.join(args.output_dir, "Clinical_metadata_classifier")
    os.makedirs(checkpoint_save_dir, exist_ok=True)
    if args.mode == "train":
        print(f"Training Linear Classifier\n")

        if args.rep_learn == "yes":
            ## Representation Learning for Age Sex
            print("Representation Learning\n")
            metadata_utils.train_loop_representation_learning(model=model, data_loader_Train=train_dataloader, data_loader_Val=val_dataloader, Fold=args.fold, output_dir=checkpoint_save_dir)

            ## Load the best model 
            state_dict = torch.load(os.path.join(checkpoint_save_dir, f"{args.model_name}", f"Fold_{args.fold}", f"{args.model_name}_Representation.pth"))
            # Load the state dictionary into the model
            model.load_state_dict(state_dict, strict=True)

            ## Freeze the model except classification head
            for i, (name, param) in enumerate(model.named_parameters()):
                if i < 10:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
                if logg:
                    print(f"Layer {i}: {name} | Size: {param.size()} | Requires_Grad: {param.requires_grad}")

        ##train the MLP model
        print("Learning classifier\n")
        metadata_utils.train_loop(model=model, data_loader_Train=train_dataloader, data_loader_Val=val_dataloader, Fold=args.fold, cls_weights=class_weights, output_dir=checkpoint_save_dir)

    else:
        print(f"Evaluating Linear Classifier")
        ### Evaluate
        results_save_dir = os.path.join(args.result_dir, "Metadata_Classifier_results")
        os.makedirs(results_save_dir, exist_ok=True)

        try:
            # Load the best state dictionary from a file
            checkpoint_load_dir = os.path.join(checkpoint_save_dir, f"{args.model_name}", f"Fold_{args.fold}")
            state_dict = torch.load(os.path.join(checkpoint_load_dir, f"{args.model_name}_Classifier.pth"))
            # Load the state dictionary into the model
            model.load_state_dict(state_dict, strict=True)

            eval_history = metadata_utils.eval_loop(model=model, data_loader_Test=test_dataloader, Class_idx_to_Label=index_to_label, Class_Label_to_idx=label_to_index, Fold=args.fold, results_dir=results_save_dir, n_classes=NUM_CLASSES, label_names=label_names)

        except:
            print("No trained MIL model found / or / Evaluation Error.")