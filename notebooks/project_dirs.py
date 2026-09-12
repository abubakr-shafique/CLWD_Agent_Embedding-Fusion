import os

current_file_path = os.path.abspath(__file__)
# print(current_file_path)
current_dir_path = os.path.dirname(os.path.abspath(__file__))
# print(current_dir_path)

# Paths
PROJ_ROOT = os.path.dirname(current_dir_path)
# print(f"PROJ_ROOT path is: {PROJ_ROOT}")

## All data paths
CODE_BASE = current_dir_path

DATA_DIR = os.path.join(PROJ_ROOT, "src", "data")
os.makedirs(DATA_DIR, exist_ok=True)

DATA_JPG_DIR = os.path.join(PROJ_ROOT, "src", "data_jpg")
os.makedirs(DATA_JPG_DIR, exist_ok=True)

CSV_DIR = os.path.join(PROJ_ROOT, "src", "csv")
os.makedirs(CSV_DIR, exist_ok=True)

RESULT_DIR = os.path.join(PROJ_ROOT, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

