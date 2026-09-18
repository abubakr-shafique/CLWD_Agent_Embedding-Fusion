import os, sys


generate_embeddings = "scripts/generate_slide_metadata_embeddings.py"
Models = ["UNI2-h", "Virchow2", "H-OPTIMUS-1"]
Folds = [0, 1, 2]

if sys.platform == "win32":
    for model_name in Models:
        for fold in Folds:
            mode = "slide"
            Command = f"python {generate_embeddings} --model_name {model_name} --mode {mode} --fold {fold}"
            os.system(Command)

    for fold in Folds:
        mode = "metadata"
        model_name = "AgeSex_Linear"
        Command = f"python {generate_embeddings} --model_name {model_name} --mode {mode} --fold {fold}"
        os.system(Command)