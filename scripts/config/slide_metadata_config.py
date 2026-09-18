import os, sys
from pathlib import Path

# Get the absolute path of the parent directory
parent_dir = Path(__file__).resolve().parent.parent

# Add the parent directory to sys.path if it's not already there
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

import project_dirs as pdir

train_csv = "CLWD_OneSlide-train.csv"
val_csv = "CLWD_OneSlide-val.csv"
test_csv = "CLWD_OneSlide-test.csv"

embedding_dir = pdir.EMBEDDINGS_DIR

target_patch_size = 224
target_magnification = 20

MIL_Model = 'ABMIL'
model_name = 'UNI2-h'

output_dir = pdir.EVAL_DIR
result_dir = pdir.RESULT_DIR

epochs= 100
batch_size = 1
lr = 1e-4
wd = 1e-4
lr_step_count = 3
loss = "CrossEntropy"
early_Stop = True
tolerance = 50

embed_dim = 1024
num_fc_layers = 2
dropout = 0.30
attn_dim = 384
gate = True