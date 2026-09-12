import os, sys
from pathlib import Path

# Get the absolute path of the parent directory
parent_dir = Path(__file__).resolve().parent.parent

# Add the parent directory to sys.path if it's not already there
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

import project_dirs as pdir

target_patch_size = 224
target_mag = 20
tissue_area = 80
patch_overlap = 1.0
smaller_level = 6
batch_size = 32

model_name = 'H-OPTIMUS-1'

csv_dir = os.path.join(pdir.CSV_DIR, "CLWD_OneSlide.csv")
data_dir = pdir.DATA_DIR
embedding_dir = pdir.EMBEDDINGS_DIR