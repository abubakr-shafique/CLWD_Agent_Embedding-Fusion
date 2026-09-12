import os, sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from sklearn.metrics import f1_score, balanced_accuracy_score

from pathlib import Path

# Get the absolute path of the parent directory
parent_dir = Path(__file__).resolve().parent.parent
# Add the parent directory to sys.path if it's not already there
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

import config.MIL_config as MIL_config
import utils.eval_utils as eval_utils

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def make_path_list(root_dir, WSI_list, file_ext=".pt"):
    full_path_list = []
    for W in WSI_list:
        full_path_list.append(os.path.join(root_dir, f"{W}{file_ext}"))

    return full_path_list


def labels_to_indices(labels, label_to_index):
    """
    Convert a list of string labels to their corresponding indices.

    Parameters
    ----------
    labels : list
        List of labels to convert.
    label_to_index : dict
        Dictionary mapping labels to integer indices.

    Returns
    -------
    list
        List of corresponding integer indices.
    """
    return [label_to_index[label] for label in labels]


def compute_class_weights(train_dataset):
    labels = torch.tensor(train_dataset.labels)
    class_counts = torch.bincount(labels)
    total_samples = len(labels)
    class_weights = total_samples / class_counts.float()
    # Normalize
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    return class_weights


class EmbeddingDataset(Dataset):
    def __init__(self, embedding_paths, coord_paths, labels, return_path=False, device='cpu'):
        """
        Args:
            embedding_paths (list of str): Paths to .pt embedding files.
            coord_paths (list of str): Paths to .npy coordinate files.
            labels (list): Corresponding labels.
            device (str): 'cpu' or 'cuda' to load tensors directly on device.
        """
        assert len(embedding_paths) == len(labels), "Embeddings and labels must be the same length."
        self.embedding_paths = embedding_paths
        self.coord_paths = coord_paths
        self.labels = labels
        self.device = device
        self.return_path = return_path

    def __len__(self):
        return len(self.embedding_paths)

    def __getitem__(self, idx):
        # Load embedding from .pt file
        embedding = torch.load(self.embedding_paths[idx], map_location=self.device) ## for pt file
        label = torch.tensor(self.labels[idx], dtype=torch.long)  # change dtype if needed
        path = self.embedding_paths[idx]
        # coords = self.coord_paths[idx]

        if self.return_path:
            return embedding, label, path
        else:
            return embedding, label

def CrossEntropy_Accuracy(outputs, labels):
    predicted_labels = torch.argmax(outputs, dim=1)

    # Compare predicted labels with ground truth labels and calculate accuracy
    correct_predictions = (predicted_labels == labels).sum().item()
    total_predictions = labels.size(0)
    accuracy = correct_predictions / total_predictions * 100.0
    
    return accuracy

def train_loop(model, data_loader_Train, data_loader_Val,
               epochs=MIL_config.epochs, lr=MIL_config.lr, 
               wd=MIL_config.wd, model_name=None,
               loss_func=MIL_config.loss, cls_weights=None,
               patch_size=MIL_config.target_patch_size, early_stop=MIL_config.early_Stop, tolerance=MIL_config.tolerance,
               input_dim=1024, output_dir=None, Fold=f"Fold_0", lr_step_count = MIL_config.lr_step_count,
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
    
    checkpoint_save_dir = os.path.join(output_dir, model_name, f"{Fold}")
    os.makedirs(checkpoint_save_dir, exist_ok=True)
    Log_file_path = os.path.join(checkpoint_save_dir, f"{model_name} Training MIL.txt")

    file = open(Log_file_path, "a")
    my_log = (f"Epochs: {epochs}, Model_Name: {model_name}, Optimizer: AdamW,\n"
              f"Loss_Func: {Loss_func},\n"
              f"Learning Rate: {lr}, Weight_Decay: {wd}. \nPatch_Size: {patch_size},\n"
              f"Batch_Size: {MIL_config.batch_size}, Early_Stop {early_stop},\n"
              f"Embed_dim: {input_dim}, WSI_dim: {MIL_config.embed_dim} \n\n")
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

            results_dict, log_dict = model(inputs, return_attention=True, return_slide_feats=True)
            
            outputs = results_dict["logits"]
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
            
            results_dict, log_dict = model(inputs, return_attention=True, return_slide_feats=True)
                
            outputs = results_dict["logits"]
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

        if val_bal_acc > Val_acc_check or val_loss < Val_loss_check: ## Best Loss ot Balanced Accuracy
        # if val_bal_acc >= Val_acc_check: ## Best Balanced Accuracy

            torch.save(model.state_dict(), os.path.join(checkpoint_save_dir, f"{model_name}_Classifier.pth"))
            print("Model Saved")
            Val_acc_check = val_bal_acc
            file = open(Log_file_path, "a")
            file.write("Model Saved \n")
            file.close()
            Val_loss_check = val_loss
            early_stopping = 0

        if current_lr > 1e-5:
            scheduler.step()

        if early_stopping >= tolerance and early_stop:
            break
    
    training_history_save_path = os.path.join(checkpoint_save_dir, f"{model_name}_training_history.npy")
    np.save(training_history_save_path, training_history)
    plot_save_path = os.path.join(checkpoint_save_dir, f"{model_name}_training_curves.png")
    eval_utils.Plot_Training_Data(train_accuracy=training_history["Train_Accuracy"], val_accuracy=training_history["Val_Accuracy"], train_loss=training_history["Train_Loss"], val_loss=training_history["Val_Loss"], save_path=plot_save_path)


def eval_loop(model, data_loader_Test, Class_idx_to_Label, Class_Label_to_idx, results_dir=None, model_name=None, Fold="Fold_0", n_classes=None, label_names=None):
    result_save_dir = os.path.join(results_dir, model_name, f"{Fold}")
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

            results_dict, log_dict = model(inputs, return_attention=True, return_slide_feats=True)
            # attention = log_dict["attention"].cpu().detach().numpy()
            
            outputs = results_dict["logits"].float()
            # accuracy = CrossEntropy_Accuracy(outputs, labels)

            predicted_labels = torch.argmax(outputs, dim=1)
            predicted_labels = predicted_labels.to("cpu").numpy()
            labels = labels.to("cpu").numpy()
            outputs = outputs.cpu().detach().numpy()
            for idx, L in enumerate(labels):
                eval_history["True_Labels"].append(Class_idx_to_Label[L])
                eval_history["Pred_Labels"].append(Class_idx_to_Label[predicted_labels[idx]])
                eval_history["Logits"].append(outputs[idx])
                eval_history["Features"].append(log_dict["slide_feats"][idx].float().cpu().detach().numpy())
            

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


def generate_slide_embeddings(model, data_loader, output_dir):

    model = model.to(device)

    for epoch in range(0, 1):
        data_loop = tqdm(data_loader)
        
        model.eval()

        for idx, (inputs, labels, path) in enumerate(data_loop):
            inputs = inputs.to(device)

            results_dict, log_dict = model(inputs, return_attention=True, return_slide_feats=True)
            
            logits = results_dict["logits"].float().cpu().detach()
            attention = log_dict["attention"].float().cpu().detach()
            attention = attention.squeeze()
            embeddings = log_dict["slide_feats"].float().cpu().detach()           

            data_loop.set_description(f"Epoch [{epoch}/{1}]")

            slide_name = os.path.basename(path[0])

            embedding_save_dir = os.path.join(output_dir, slide_name)
            torch.save(embeddings, embedding_save_dir)


            attention_save_dir = output_dir.replace("\\slide\\", "\\slide_attention\\")
            os.makedirs(attention_save_dir, exist_ok=True)
            torch.save(attention, os.path.join(attention_save_dir, slide_name))