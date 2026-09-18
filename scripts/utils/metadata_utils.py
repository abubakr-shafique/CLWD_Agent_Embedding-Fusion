import os, sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report

from pathlib import Path

# Get the absolute path of the parent directory
parent_dir = Path(__file__).resolve().parent.parent
# Add the parent directory to sys.path if it's not already there
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

import config.metadata_classifier_config as Meta_config
import utils.eval_utils as eval_utils

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class ClinicalDataset(Dataset):

    def __init__(self, df, label_to_index, sex_to_index, has_labels=True, get_WSI_ID = False):

        self.df = df.reset_index(drop=True)
        self.has_labels = has_labels
        self.get_WSI_ID = get_WSI_ID

        # ----------------------------------------------------
        # Age normalization
        # Age is explicitly normalized from 0-100 to 0-1
        # ----------------------------------------------------
        age = self.df["Age"].astype(float).values
        min_age = 24.0#np.min(age)
        max_age = 80.0#np.max(age)
        age = [(ag - min_age) / (max_age - min_age) for ag in age]

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

        if has_labels:
            self.y = (
                self.df["Benchmark_Label_7class"]
                .map(label_to_index)
                .values
                .astype(np.int64)
            )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        x = torch.tensor(self.X[idx], dtype=torch.float32)

        if self.has_labels:
            y = torch.tensor(self.y[idx], dtype=torch.long)
            if self.get_WSI_ID:
                return x, y, self.WSI_ID[idx]
            else:
                return x, y

        return x

class AgeSexClassifier(nn.Module):
    def __init__(self, input = 2, layers = [64, 128, 384], dropout=0.3, num_classes=7):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input, layers[0]),
            nn.ReLU(),
            nn.LayerNorm(layers[0]),

            nn.Linear(layers[0], layers[1]),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(layers[1], layers[2]),
        )
        self.reconstruct = nn.Linear(layers[2], input)
        self.classifier = nn.Linear(layers[2], num_classes)

    def forward(self, x):

        Embeds = self.network(x)
        age_Sex = self.reconstruct(Embeds)
        logits = self.classifier(Embeds)


        return logits, Embeds, age_Sex


def CrossEntropy_Accuracy(outputs, labels):
    predicted_labels = torch.argmax(outputs, dim=1)

    # Compare predicted labels with ground truth labels and calculate accuracy
    correct_predictions = (predicted_labels == labels).sum().item()
    total_predictions = labels.size(0)
    accuracy = correct_predictions / total_predictions * 100.0
    
    return accuracy

def train_loop_representation_learning(model, data_loader_Train, data_loader_Val,
               epochs=Meta_config.epochs, lr=Meta_config.lr, 
               wd=Meta_config.wd, model_name= Meta_config.model_name,
               early_stop=Meta_config.early_Stop, tolerance=Meta_config.tolerance,
               output_dir=None, Fold=0, lr_step_count=Meta_config.lr_step_count,
               ):
    
    early_stopping = 0
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.MSELoss()
    Loss_func = "MSE Loss"

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_count, gamma=0.1)
    
    checkpoint_save_dir = os.path.join(output_dir, model_name, f"Fold_{Fold}")
    os.makedirs(checkpoint_save_dir, exist_ok=True)
    Log_file_path = os.path.join(checkpoint_save_dir, f"{model_name} Representation Learning.txt")

    file = open(Log_file_path, "a")
    my_log = (f"Epochs: {epochs}, Model_Name: {model_name}, Optimizer: AdamW,\n"
              f"Loss_Func: {Loss_func},\n"
              f"Learning Rate: {lr}, Weight_Decay: {wd}. \n"
              f"Batch_Size: {Meta_config.batch_size}, Early_Stop {early_stop},\n"
                )
    file.write(my_log)
    file.close()

    training_history = {
        "epoch": [],
        "Train_Loss": [],
        "Val_Loss": []
        }
    
    Val_loss_check = np.inf
    for epoch in range(0, epochs):
        data_loop = tqdm(data_loader_Train)

        training_history["epoch"].append(epoch)

        Avg_loss = []

        for idx, (inputs, labels) in enumerate(data_loop):
            inputs = inputs.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            logits, Embeds, age_Sex = model(inputs)            
            outputs = age_Sex

            loss = criterion(outputs, inputs)

            Avg_loss.append(loss.item())
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            current_lr = optimizer.param_groups[0]["lr"]
            
            data_loop.set_description(f"Epoch [{epoch}/{epochs}]")
            data_loop.set_postfix(Loss = loss.item(), Lr = current_lr)

        train_loss = np.round(np.mean(Avg_loss), 3)
        print('Avg Train Loss: {:.2f}'.format(train_loss))
        training_history["Train_Loss"].append(train_loss)

        file = open(Log_file_path, "a")
        my_log = f"Epoch: {epoch}, Avg_Train_Loss: {train_loss} \n" 
        file.write(my_log)
        file.close()


        Avg_loss_val = []
        
        for inputs, labels in data_loader_Val:
            inputs = inputs.to(device)

            
            logits, Embeds, age_Sex = model(inputs)            
            outputs = age_Sex

            loss_val = criterion(outputs, inputs)
            
            Avg_loss_val.append(loss_val.item())
    
        val_loss = np.round(np.mean(Avg_loss_val), 3)
        print('Avg Val Loss: {:.2f}'.format(val_loss))
        training_history["Val_Loss"].append(val_loss)
        
        file = open(Log_file_path, "a")
        my_log = f"Epoch: {epoch}, Avg_Val_Loss: {val_loss}\n" 
        file.write(my_log)
        file.close()

        early_stopping = early_stopping + 1

        if val_loss < Val_loss_check: ## Best Loss

            torch.save(model.state_dict(), os.path.join(checkpoint_save_dir, f"{model_name}_Representation.pth"))
            print("Model Saved")
            file = open(Log_file_path, "a")
            file.write("Model Saved \n")
            file.close()
            Val_loss_check = val_loss
            early_stopping = 0

        if current_lr > 1e-6:
            scheduler.step()

        if early_stopping >= tolerance and early_stop:
            break
    
    training_history_save_path = os.path.join(checkpoint_save_dir, f"{model_name}_training_history_representation_learning.npy")
    np.save(training_history_save_path, training_history)


def train_loop(model, data_loader_Train, data_loader_Val,
               epochs=Meta_config.epochs, lr=Meta_config.lr, 
               wd=Meta_config.wd, model_name= Meta_config.model_name,
               loss_func=Meta_config.loss, cls_weights=None, early_stop=Meta_config.early_Stop, tolerance=Meta_config.tolerance,
               output_dir=None, Fold=0, lr_step_count=Meta_config.lr_step_count,
               ):
    
    early_stopping = 0
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if loss_func == "CrossEntropy" and cls_weights == None:
        criterion = nn.CrossEntropyLoss()
        Loss_func = "CrossEntropy"
    elif loss_func == "CrossEntropy" and cls_weights is not None:
        cls_weights = cls_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=cls_weights)
        Loss_func = f"Weighted CrossEntropy: {cls_weights}"

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_count, gamma=0.1)
    
    checkpoint_save_dir = os.path.join(output_dir, model_name, f"Fold_{Fold}")
    os.makedirs(checkpoint_save_dir, exist_ok=True)
    Log_file_path = os.path.join(checkpoint_save_dir, f"{model_name} Training Metadata MLP.txt")

    file = open(Log_file_path, "a")
    my_log = (f"Epochs: {epochs}, Model_Name: {model_name}, Optimizer: AdamW,\n"
              f"Loss_Func: {Loss_func},\n"
              f"Learning Rate: {lr}, Weight_Decay: {wd}. \n"
              f"Batch_Size: {Meta_config.batch_size}, Early_Stop {early_stop},\n"
            )
    file.write(my_log)
    file.close()

    training_history = {
        "epoch": [],
        "Train_Accuracy": [],
        "Train_Loss": [],
        "Val_Accuracy": [],
        "Val_Loss": []
        }
    
    Val_acc_check = 0
    Val_loss_check = np.inf
    for epoch in range(0, epochs):
        data_loop = tqdm(data_loader_Train)

        training_history["epoch"].append(epoch)

        Avg_loss = []
        Avg_accu = []
        train_Label = []
        train_pred = []

        for idx, (inputs, labels) in enumerate(data_loop):
            inputs = inputs.to(device)
            for L in labels:
                train_Label.append(L)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            logits, Embeds, age_Sex = model(inputs)            
            outputs = logits

            predicted_labels = torch.argmax(outputs, dim=1).cpu()
            for P in predicted_labels:
                train_pred.append(P)
            loss = criterion(outputs, labels)
                
            accuracy = CrossEntropy_Accuracy(outputs, labels)

            Avg_loss.append(loss.item())
            Avg_accu.append(accuracy)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            current_lr = optimizer.param_groups[0]["lr"]
            
            data_loop.set_description(f"Epoch [{epoch}/{epochs}]")
            data_loop.set_postfix(Accuracy = accuracy, Loss = loss.item(), Lr = current_lr)
        
        train_accuracy = np.round(np.mean(Avg_accu), 3)
        train_loss = np.round(np.mean(Avg_loss), 3)
        train_bal_acc = np.round(balanced_accuracy_score(np.array(train_Label), np.array(train_pred)) * 100, 3) 
        # print('Avg Train Accuracy: {:.2f}%'.format(train_accuracy))
        print('Train Bal. Accuracy: {:.2f}%'.format(train_bal_acc))
        print('Avg Train Loss: {:.2f}'.format(train_loss))
        
        training_history["Train_Accuracy"].append(train_accuracy)
        training_history["Train_Loss"].append(train_loss)

        file = open(Log_file_path, "a")
        my_log = f"Epoch: {epoch}, Avg_Train_Loss: {train_loss}, Avg_Train_Accuracy: {train_accuracy}, 'Train Balance Accuracy: {train_bal_acc} \n" 
        file.write(my_log)
        file.close()


        Avg_loss_val = []
        Avg_accu_val = []
        val_Label = []
        val_pred = []
        
        for inputs, labels in data_loader_Val:
            
            inputs = inputs.to(device)
            for L in labels:
                val_Label.append(L)
            labels = labels.to(device)
            
            logits, Embeds, age_Sex = model(inputs)            
            outputs = logits

            predicted_labels = torch.argmax(outputs, dim=1).cpu()
            for P in predicted_labels:
                val_pred.append(P)
            loss_val = criterion(outputs, labels)
         
            accuracy_val = CrossEntropy_Accuracy(outputs, labels)
            
            Avg_loss_val.append(loss_val.item())
            Avg_accu_val.append(accuracy_val)
    

        val_accuracy = np.round(np.mean(Avg_accu_val), 3)
        val_loss = np.round(np.mean(Avg_loss_val), 3)
        val_bal_acc = np.round(balanced_accuracy_score(np.array(val_Label), np.array(val_pred)) * 100, 3)
        # print('Avg Val Accuracy: {:.2f}%'.format(val_accuracy))
        print('Val Bal. Accuracy: {:.2f}%'.format(val_bal_acc))
        print('Avg Val Loss: {:.2f}'.format(val_loss))

        training_history["Val_Accuracy"].append(val_accuracy)
        training_history["Val_Loss"].append(val_loss)
        
        file = open(Log_file_path, "a")
        my_log = f"Epoch: {epoch}, Avg_Val_Loss: {val_loss}, Avg_Val_Accuracy: {val_accuracy}, Val Balance Accuracy: {val_bal_acc} \n" 
        file.write(my_log)
        file.close()

        early_stopping = early_stopping + 1

        # if val_bal_acc > Val_acc_check or val_loss < Val_loss_check: ## Best Loss ot Balanced Accuracy
        # if val_loss < Val_loss_check: ## Best Loss
        if val_bal_acc > Val_acc_check: ## Best Balanced Accuracy

            torch.save(model.state_dict(), os.path.join(checkpoint_save_dir, f"{model_name}_Classifier.pth"))
            print("Model Saved")
            Val_acc_check = val_bal_acc
            file = open(Log_file_path, "a")
            file.write("Model Saved \n")
            file.close()
            Val_loss_check = val_loss
            early_stopping = 0

        if current_lr > 1e-6:
            scheduler.step()

        if early_stopping >= tolerance and early_stop:
            break
    
    training_history_save_path = os.path.join(checkpoint_save_dir, f"{model_name}_training_history.npy")
    np.save(training_history_save_path, training_history)
    plot_save_path = os.path.join(checkpoint_save_dir, f"{model_name}_training_curves.png")
    eval_utils.Plot_Training_Data(train_accuracy=training_history["Train_Accuracy"], val_accuracy=training_history["Val_Accuracy"], train_loss=training_history["Train_Loss"], val_loss=training_history["Val_Loss"], save_path=plot_save_path)

def eval_loop(model, data_loader_Test, Class_idx_to_Label, Class_Label_to_idx, results_dir=None, model_name=Meta_config.model_name, Fold=0, n_classes=None, label_names=None):
    result_save_dir = os.path.join(results_dir, model_name, f"Fold_{Fold}")
    os.makedirs(result_save_dir, exist_ok=True)

    model = model.to(device)

    eval_history = {
    "Pred_Labels": [],
    "True_Labels": [],
    "Logits": [],
    "Features": [],
    }

    for epoch in range(0, 1):
        data_loop = tqdm(data_loader_Test)
        
        model.eval()

        for idx, (inputs, labels) in enumerate(data_loop):
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits, Embeds, age_Sex = model(inputs)            
            outputs = logits

            predicted_labels = torch.argmax(outputs, dim=1)
            predicted_labels = predicted_labels.to("cpu").numpy()
            labels = labels.to("cpu").numpy()
            outputs = outputs.cpu().detach().numpy()
            for idx, L in enumerate(labels):
                eval_history["True_Labels"].append(Class_idx_to_Label[L])
                eval_history["Pred_Labels"].append(Class_idx_to_Label[predicted_labels[idx]])
                eval_history["Logits"].append(outputs[idx])
                eval_history["Features"].append(Embeds.float().cpu().detach().numpy())
            

            data_loop.set_description(f"Epoch [{epoch}/{1}]")
        

    test_accuracy = np.round(balanced_accuracy_score(np.array(eval_history["True_Labels"]), np.array(eval_history["Pred_Labels"])) * 100, 3)
    print('Test Bal. Accuracy: {:.2f}%'.format(test_accuracy))
    

    eval_history_save_path = os.path.join(result_save_dir, f"{model_name}_eval.npy")
    np.save(eval_history_save_path, eval_history)

    ## save in this folder
    results_summary_save_dir = os.path.join(result_save_dir, f"{model_name}_results.npy")
    results = eval_utils.multiclass_classification_report(y_true=eval_history["True_Labels"], y_pred=eval_history["Pred_Labels"], n_classes=n_classes, class_names=label_names, save_dir=results_summary_save_dir)

    ROC_save_dir = os.path.join(result_save_dir, f"{model_name}_ROC.jpg")
    eval_utils.plot_multiclass_roc(eval_history, classes=Class_idx_to_Label, class_to_idx=Class_Label_to_idx, output_file=ROC_save_dir)
    
    accuracy_save_dir = os.path.join(result_save_dir, f"{model_name}_Overall_Accuracy.jpg")
    metrics = eval_utils.evaluate_and_plot_metrics(eval_history=eval_history, class_to_idx=Class_Label_to_idx, output_file=accuracy_save_dir)
    
    return eval_history

def generate_metadata_embeddings(model, metadata_loader, output_dir):

    model = model.to(device)

    for epoch in range(0, 1):
        data_loop = tqdm(metadata_loader)
        
        model.eval()

        for idx, (inputs, labels, WSI_IDs) in enumerate(data_loop):
            inputs = inputs.to(device)

            logits, Embeds, age_Sex = model(inputs)            

            Embeds = Embeds.float().cpu().detach()
            for i, E in enumerate(Embeds):
                WSI_ID = WSI_IDs[i]
                embedding_save_dir = os.path.join(output_dir, f"{WSI_ID}.pt")
                torch.save(E, embedding_save_dir)
                