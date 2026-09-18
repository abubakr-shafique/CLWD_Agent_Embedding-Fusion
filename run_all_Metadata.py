import os, sys


script_path = "scripts/clinical_metadata_classifier.py"
Folds = [0, 1, 2]

if sys.platform == "win32":

    for fold in Folds:
        print(f"Fold: {fold}\n")
        mode = "train"
        Command = f"python {script_path} --mode {mode} --fold {fold}"
        os.system(Command)

        mode = "eval"
        Command = f"python {script_path} --mode {mode} --fold {fold}"
        os.system(Command)