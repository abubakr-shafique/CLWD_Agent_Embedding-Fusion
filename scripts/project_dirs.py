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
os.makedirs(DATA_DIR, exist_ok=True)

EMBEDDINGS_DIR = os.path.join(PROJ_ROOT, "src", "embeddings")
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

EVAL_DIR = os.path.join(PROJ_ROOT, "src", "eval")
os.makedirs(EVAL_DIR, exist_ok=True)

AGENT_DIR = os.path.join(PROJ_ROOT, "src", "agent")
os.makedirs(AGENT_DIR, exist_ok=True)

CSV_DIR = os.path.join(PROJ_ROOT, "src", "csv")
os.makedirs(CSV_DIR, exist_ok=True)

PATCH_COORDS_DIR = os.path.join(PROJ_ROOT, "src", "patch_coords")
os.makedirs(PATCH_COORDS_DIR, exist_ok=True)

THUMBNAIL_DIR = os.path.join(PROJ_ROOT, "src", "thumbnail")
os.makedirs(THUMBNAIL_DIR, exist_ok=True)

VISUALIZATION_DIR = os.path.join(PROJ_ROOT, "src", "patch_visualization")
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

PATCH_IMAGES_DIR = os.path.join(PROJ_ROOT, "src", "patch_images")
os.makedirs(PATCH_IMAGES_DIR, exist_ok=True)

ARTIFACTS_DIR = os.path.join(PROJ_ROOT, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

RESULT_DIR = os.path.join(PROJ_ROOT, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

