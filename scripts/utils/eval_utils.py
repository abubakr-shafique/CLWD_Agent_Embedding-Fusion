import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, classification_report, accuracy_score, roc_auc_score, roc_curve, auc, balanced_accuracy_score
from sklearn.preprocessing import label_binarize
import pandas as pd
import seaborn as sns

def plot_confusion_matrix_norm(cm, title, output_file, class_names, Figsize=(16, 10),
                               cellFontSize=14,lineSpace=2.0,fontsizelabel=14,
                               xLabelsRotation=0, tickFontSize=12):

    # Normalize by row sums
    cm_norm_by_row = cm / cm.sum(axis=1, keepdims=True)
    # Convert to percentage
    cm_percentage = np.vectorize(lambda v: f'{v:.1%}')(cm_norm_by_row)
    plt.figure(figsize=Figsize)
    # plt.figure(figsize=(9, 7))
    # Use a larger font size for the tick labels and annotation text

    sns.heatmap(
        cm_norm_by_row,
        annot=cm_percentage,
        fmt='',
        cmap='Blues',  # Use the custom colormap
        annot_kws={'size': cellFontSize, 'fontweight': 'bold'},  # Make the cell font bold
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=lineSpace,  # Add visible separation between cells
        linecolor='white',  # Set the color of the cell separators to white
        vmin=-0.05,  # Set the minimum value for the color scale
        vmax=1.0   # Set the maximum value for the color scale
    )
    plt.title(title, fontsize=fontsizelabel)  # Adjust the font size for the title
    plt.xlabel('Model Diagnosis', fontsize=fontsizelabel)  # Adjust the font size for labels
    plt.ylabel('True Diagnosis', fontsize=fontsizelabel)  # Adjust the font size for labels
    # Increase the font size for tick labels
    tick_label_font = {'fontsize': tickFontSize, 'weight': 'bold'}
    plt.xticks(rotation=xLabelsRotation)
    plt.yticks(rotation=0)
    # Make tick labels bold

    plt.gca().xaxis.set_ticklabels(class_names, fontdict=tick_label_font)
    plt.gca().yaxis.set_ticklabels(class_names, fontdict=tick_label_font)

    # Ensure that the axis labels are fully visible
    plt.tight_layout()
    # Save the confusion matrix plot to the output directory
    if title is not None:
        plt.savefig(output_file, dpi=300)
    plt.close()

def plot_sensitivity_specificity(sensitivity: dict, specificity: dict, title: str = "Sensitivity and Specificity", output_file=None ):

    classes = list(sensitivity.keys())
    sens_values = [float(sensitivity[c]) for c in classes]
    spec_values = [float(specificity[c]) for c in classes]

    x = np.arange(len(classes))
    width = 0.35

    plt.figure(figsize=(12, 8))

    bars1 = plt.bar(x - width / 2, sens_values, width, label="Sensitivity")
    bars2 = plt.bar(x + width / 2, spec_values, width, label="Specificity")

    # ---- Add text labels ----
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=10
            )

    plt.xticks(x, classes, rotation=0)
    plt.ylim(0.1, 1.1)
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    if output_file is not None:
        plt.savefig(output_file, dpi=300)
    # plt.show()
    plt.close()


def Plot_Training_Data(train_accuracy, val_accuracy, train_loss, val_loss, save_path=None):
    
    # Epochs
    epochs = np.arange(1, len(train_accuracy)+1)

    train_accuracy = np.array(train_accuracy)
    val_accuracy = np.array(val_accuracy)

    train_loss = np.array(train_loss)
    val_loss = np.array(val_loss)
    # Create subplots: 1 row, 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # -------- Accuracy subplot --------
    axes[0].plot(epochs, train_accuracy, label='Train Accuracy')
    axes[0].plot(epochs, val_accuracy, label='Validation Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Training vs Validation Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    # -------- Loss subplot --------
    axes[1].plot(epochs, train_loss, label='Train Loss')
    axes[1].plot(epochs, val_loss, label='Validation Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Training vs Validation Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    # Save figure
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    # plt.show()
    plt.close()


def multiclass_classification_report(y_true, y_pred, n_classes=7, class_names=None, save_dir=None):

    results = {}

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]
    
    # --------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------
    cm = confusion_matrix(y_true, y_pred)
    results["confusion_matrix"] = cm
    plot_confusion_matrix_norm(cm, title="Classification Confusion Matrix", output_file=f"{save_dir[:-4]}_CM.jpg", class_names=class_names)
    # --------------------------------------------------
    # Precision / Recall / F1
    # --------------------------------------------------
    results["precision_macro"] = precision_score(y_true, y_pred, average="macro")
    results["recall_macro"] = recall_score(y_true, y_pred, average="macro")
    results["f1_macro"] = f1_score(y_true, y_pred, average="macro")

    results["precision_micro"] = precision_score(y_true, y_pred, average="micro")
    results["recall_micro"] = recall_score(y_true, y_pred, average="micro")
    results["f1_micro"] = f1_score(y_true, y_pred, average="micro")

    results["precision_weighted"] = precision_score(y_true, y_pred, average="weighted")
    results["recall_weighted"] = recall_score(y_true, y_pred, average="weighted")
    results["f1_weighted"] = f1_score(y_true, y_pred, average="weighted")

    results["classification_report"] = classification_report(y_true, y_pred)
    results["balanced_accuracy"] = np.round(balanced_accuracy_score(np.array(y_true), np.array(y_pred)) * 100, 2)
    balanced_accuracy = results["balanced_accuracy"]
    
    # print(results["classification_report"])
    CR_save_path = f"{save_dir[:-4]}_CR.txt"
    text_file = open(CR_save_path, "w")
    text_file.write(results["classification_report"])
    text_file.write("\n\n")
    text_file.write(f"Balanced Accuracy: {balanced_accuracy}")
    text_file.close()
    # --------------------------------------------------
    # Sensitivity & Specificity (per class)
    # --------------------------------------------------
    sensitivity = {}
    specificity = {}

    for i, cls in enumerate(class_names):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - (TP + FP + FN)

        sensitivity[cls] = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        specificity[cls] = TN / (TN + FP) if (TN + FP) > 0 else 0.0

    rounded_sensitivity = {k: round(v, 2) for k, v in sensitivity.items()}
    rounded_specificity = {k: round(v, 2) for k, v in specificity.items()}

    results["sensitivity"] = rounded_sensitivity
    results["specificity"] = rounded_specificity

    plot_sensitivity_specificity(sensitivity=rounded_sensitivity, specificity=rounded_specificity, output_file=f"{save_dir[:-4]}_Sensitivity-Specificity.jpg")

    # print(f"Sensitivity: {rounded_sensitivity}")
    # print(f"Specificity: {rounded_specificity}")


    if save_dir is not None:
        np.save(save_dir, results)

    return results

def evaluate_and_plot_metrics(eval_history, class_to_idx=None, output_file=None):

    # 1. Prepare data
    y_true_names = np.array(eval_history["True_Labels"])
    logits = np.array(eval_history["Logits"])
    
    y_true = []
    if class_to_idx is not None:
        for N in y_true_names:
            y_true.append(class_to_idx[N])
    y_true = np.array(y_true)


    # Convert logits to probabilities for AUC-ROC
    y_probs = softmax(logits, axis=1)
    # Get class predictions (0, 1, or 2) for Accuracy and F1
    y_pred = np.argmax(logits, axis=1)
    
    # 2. Calculate Metrics
    # multi_class='ovr' (One-vs-Rest) is standard for multi-class AUC
    if len(class_to_idx) <=2:
        auc_roc = roc_auc_score(y_true, y_probs[:,1], multi_class='ovr', average='macro')
    else:
        auc_roc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    metrics = {
        "AUC-ROC": auc_roc,
        "Accuracy": acc,
        "F1 (Macro)": f1_macro,
        "F1 (Weighted)": f1_weighted
    }
    
    # 3. Plotting
    names = list(metrics.keys())
    values = list(metrics.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, values)#, color=['#4285F4', '#EA4335', '#FBBC05', '#34A853'])
    
    # Add text labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 3), ha='center', va='bottom', fontsize=14)
    
    plt.ylim(0, 1.1) # Metrics are between 0 and 1
    plt.ylabel('Score')
    plt.title('Model Evaluation Metrics')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    if output_file is not None:
        plt.savefig(output_file, dpi=300)
    # plt.show()
    plt.close()


    return metrics

def plot_multiclass_roc(eval_history, classes, class_to_idx, output_file=None):
    true_labels_class = np.array(eval_history['True_Labels'])
    logits = np.array(eval_history['Logits'])


    true_labels = np.array([class_to_idx[L] for L in true_labels_class])
    n_classes = len(classes)

    if logits.ndim == 1:
        probs = 1 / (1 + np.exp(-logits))
        probs = np.stack([1 - probs, probs], axis=1)
    else:
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)

    y_true_bin = label_binarize(true_labels, classes=list(range(n_classes)))
    if n_classes == 2 and y_true_bin.shape[1] == 1:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Overall AUC — micro-average (pools all samples/classes together)
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Overall AUC — macro-average (averages per-class TPR on common grid)
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize=(7, 6))
    colors = plt.cm.tab10.colors
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], color=colors[i % len(colors)], lw=3,
                 label=f"{classes[i]} (AUC = {roc_auc[i]:.3f})")

    # plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', lw=3,
    #           label=f"Micro-average (AUC = {roc_auc['micro']:.3f})")
    plt.plot(fpr["macro"], tpr["macro"], color='navy', linestyle=':', lw=3,
              label=f"Macro-average (AUC = {roc_auc['macro']:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('One-vs-Rest ROC Curve with Overall AUC')
    plt.legend(loc='lower right')
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=200)
    plt.close()

    return roc_auc

def plot_multiclass_roc_agent(eval_history, classes, output_file=None):

    true_labels = eval_history["True_label"].numpy()
    logits = eval_history["logits"].numpy()
    n_classes = len(classes)

    if logits.ndim == 1:
        probs = 1 / (1 + np.exp(-logits))
        probs = np.stack([1 - probs, probs], axis=1)
    else:
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)

    y_true_bin = label_binarize(true_labels, classes=list(range(n_classes)))
    if n_classes == 2 and y_true_bin.shape[1] == 1:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Overall AUC — micro-average (pools all samples/classes together)
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Overall AUC — macro-average (averages per-class TPR on common grid)
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize=(7, 6))
    colors = plt.cm.tab10.colors
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], color=colors[i % len(colors)], lw=3,
                 label=f"{classes[i]} (AUC = {roc_auc[i]:.3f})")

    # plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', lw=3,
    #           label=f"Micro-average (AUC = {roc_auc['micro']:.3f})")
    plt.plot(fpr["macro"], tpr["macro"], color='navy', linestyle=':', lw=3,
              label=f"Macro-average (AUC = {roc_auc['macro']:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('One-vs-Rest ROC Curve with Overall AUC')
    plt.legend(loc='lower right')
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=200)
    plt.close()

    return roc_auc

def evaluate_and_plot_metrics_agent(eval_history, class_to_idx=None, output_file=None):

    # 1. Prepare data
    y_true = eval_history["True_label"].numpy()
    logits = eval_history["logits"].numpy()


    # Convert logits to probabilities for AUC-ROC
    y_probs = softmax(logits, axis=1)
    # Get class predictions (0, 1, or 2) for Accuracy and F1
    y_pred = np.argmax(logits, axis=1)
    
    # 2. Calculate Metrics
    # multi_class='ovr' (One-vs-Rest) is standard for multi-class AUC
    if len(class_to_idx) <=2:
        auc_roc = roc_auc_score(y_true, y_probs[:,1], multi_class='ovr', average='macro')
    else:
        auc_roc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    metrics = {
        "AUC-ROC": auc_roc,
        "Accuracy": acc,
        "F1 (Macro)": f1_macro,
        "F1 (Weighted)": f1_weighted
    }
    
    # 3. Plotting
    names = list(metrics.keys())
    values = list(metrics.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, values)#, color=['#4285F4', '#EA4335', '#FBBC05', '#34A853'])
    
    # Add text labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 3), ha='center', va='bottom', fontsize=14)
    
    plt.ylim(0, 1.1) # Metrics are between 0 and 1
    plt.ylabel('Score')
    plt.title('Model Evaluation Metrics')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    if output_file is not None:
        plt.savefig(output_file, dpi=300)
    # plt.show()
    plt.close()


    return metrics