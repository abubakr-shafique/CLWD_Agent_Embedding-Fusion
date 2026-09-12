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

import config.MIL_config as MIL_config
import utils.eval_utils as eval_utils

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Slide_Clinical_Dataset(Dataset):

    def __init__(self, df, label_to_index, sex_to_index, embedding_root, file_ext=".pt", device='cpu'):

        self.df = df.reset_index(drop=True)
        self.embedding_root = embedding_root
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

        Age_sex = torch.tensor(self.X[idx], dtype=torch.float32)
        embedding_path = os.path.join(self.embedding_root, f"{self.WSI_ID[idx]}{self.file_ext}")
        embedding = torch.load(embedding_path, map_location=self.device) ## for pt file

        label = torch.tensor(self.y[idx], dtype=torch.long)

        

        return Age_sex, embedding, label

class AgeSexClassifier(nn.Module):
    def __init__(self, layers=[64, 128, 384], num_classes=7):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2, layers[0]),
            nn.ReLU(),
            nn.LayerNorm(layers[0]),

            nn.Linear(layers[0], layers[1]),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(layers[1], layers[2]),
        )
        self.classifier = nn.Linear(layers[2], num_classes)

    def forward(self, x):

        Embeds = self.network(x)
        logits = self.classifier(Embeds)

        return logits, Embeds

class Slide_MetaData_Classifier(nn.Module):
    def __init__(self, slide_model=None, metadata_model=None, slide_embed = 1024, meta_embed = 384, num_classes=7, dropout=0.2):
        super().__init__()
        self.slide_model = slide_model
        self.metadata_model = metadata_model

        self.network = nn.Sequential(
            nn.Linear((slide_embed+meta_embed), slide_embed),
            nn.ReLU(),
            nn.LayerNorm(slide_embed),

            nn.Linear(slide_embed, slide_embed),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(slide_embed, slide_embed),
        )
        self.classifier = nn.Linear(slide_embed, num_classes)

    def forward(self, AgeSex, Embeddings):

        meta_logits, meta_Embeds = self.metadata_model(AgeSex)

        results_dict, log_dict = self.slide_model(Embeddings, return_attention=True, return_slide_feats=True)
        slide_Embeds = log_dict["slide_feats"]

        combined_embedding = torch.cat((meta_Embeds, slide_Embeds), dim=1)

        Embeds= self.network(combined_embedding)
        logits = self.classifier(Embeds)

        return logits, Embeds

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

        for idx, (Age_sex, embedding, labels) in enumerate(data_loop):
            
            Age_sex = Age_sex.to(device)
            embedding = embedding.to(device)
            for L in labels:
                train_Label.append(L)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            logits, Embeds = model(Age_sex, embedding)
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
        
        for Age_sex, embedding, labels in data_loader_Val:
            
            Age_sex = Age_sex.to(device)
            embedding = embedding.to(device)
            for L in labels:
                val_Label.append(L)
            labels = labels.to(device)
            
            logits, Embeds = model(Age_sex, embedding)
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
        # if val_loss < Val_loss_check: ## Best Loss ot Balanced Accuracy
        if val_bal_acc > Val_acc_check: ## Best Loss ot Balanced Accuracy

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

        for idx, (Age_sex, embedding, labels) in enumerate(data_loop):
            Age_sex = Age_sex.to(device)
            embedding = embedding.to(device)
            labels = labels.to(device)

            logits, Embeds = model(Age_sex, embedding)
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