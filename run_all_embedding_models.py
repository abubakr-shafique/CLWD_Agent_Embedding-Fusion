import os, sys


generate_embeddings = "scripts/generate_embeddings.py"
Models = ["UNI2-h", "H-OPTIMUS-1", "Virchow2"]

if sys.platform == "win32":
    for model_name in Models:
        Command = f"python {generate_embeddings} --model_name {model_name}"
        os.system(Command)