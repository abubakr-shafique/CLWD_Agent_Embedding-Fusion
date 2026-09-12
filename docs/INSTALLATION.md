# Environment Setup Guide

## One step environment setup for windows:

From the root directory of the project ```./CLWD_Embedding-Fusion/```, open Anaconda Prompt, and run the command:

```
python setup_environment.py
```

[setup_environment.py](https://github.com/abubakr-shafique/CLWD_Embedding-Fusion/blob/main/setup_environment.py) automatically creates a new virtual environment with the name `CLWD` using [environment.yml](https://github.com/abubakr-shafique/CLWD_Embedding-Fusion/blob/main/environment.yml) file, and also install torch, torchvision with cuda support and all other required libraries from the [requirements.txt](https://github.com/abubakr-shafique/CLWD_Embedding-Fusion/blob/main/requirements.txt) with in the newly created virtual environment.


## Step by Step environment setup guide:
### Step1:
From the root directory of the project ```./CLWD_Embedding-Fusion/```, open Anaconda Prompt, and run the following command to create a virtual environemnt:
```
conda env create -f environment.yml
```
### Step2:
Activate the newly created virtual environment, ```CLWD```
```
conda activate CLWD
```

### Step3:
Within the newly activated virtual environment, install torch 2.9.1 and torchvision 0.24.1 with cuda 12.8 support with the following command:
```
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
```

### Step4:
Finally, install the remaining required libraries for the project:
```
pip install -r requirements.txt
```

Your Virtual environment is ready for the project.

<br>
<br>

[Return to the main page](https://github.com/abubakr-shafique/CLWD_Embedding-Fusion/tree/main)