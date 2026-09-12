import os, sys
from pathlib import Path

# Get the absolute path of the parent directory
parent_dir = Path(__file__).resolve().parent.parent

# Add the parent directory to sys.path if it's not already there
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

import project_dirs as pdir

output_dir = pdir.EVAL_DIR
result_dir = pdir.RESULT_DIR

model_name = "AgeSex_Linear"

fold = 2

train_csv = "CLWD_OneSlide-train.csv"
val_csv = "CLWD_OneSlide-val.csv"
test_csv = "CLWD_OneSlide-test.csv"

batch_size = 32
lr=1e-4
wd=1e-4
epochs = 200
loss = "CrossEntropy"

early_Stop=True
tolerance=50
lr_step_count = 50