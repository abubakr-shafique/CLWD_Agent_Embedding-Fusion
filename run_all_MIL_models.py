import os, sys


generate_embeddings = "scripts/MIL_wsi_encoder.py"
Models = ["UNI2-h", "H-OPTIMUS-1", "Virchow2"]
Folds = [0, 1, 2]

if sys.platform == "win32":

    for model_name in Models:
        for fold in Folds:
            print(f"{model_name} for Fold: {fold}\n")
            mode = "train"
            Command = f"python {generate_embeddings} --model_name {model_name} --mode {mode} --fold {fold}"
            os.system(Command)

            mode = "eval"
            Command = f"python {generate_embeddings} --model_name {model_name} --mode {mode} --fold {fold}"
            os.system(Command)