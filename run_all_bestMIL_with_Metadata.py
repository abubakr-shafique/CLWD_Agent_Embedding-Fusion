import os, sys


script_path = "scripts/Slide_with_Metadata_classifier.py"
Models = ["H-OPTIMUS-1"]
Folds = [0, 1, 2]

if sys.platform == "win32":

    for model_name in Models:
        for fold in Folds:
            print(f"{model_name} for Fold: {fold}\n")
            mode = "train"
            Command = f"python {script_path} --model_name {model_name} --mode {mode} --fold {fold}"
            os.system(Command)

            mode = "eval"
            Command = f"python {script_path} --model_name {model_name} --mode {mode} --fold {fold}"
            os.system(Command)