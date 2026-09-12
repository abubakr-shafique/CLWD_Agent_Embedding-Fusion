import os, sys


if sys.platform == "win32":
    current_path = os.getcwd()
    create_environment_path = os.path.join(current_path, "environment.yml")

    ## create environment
    Command = f"conda env create -f {create_environment_path}"
    os.system(Command)

    venv_path = os.path.join(sys.prefix, "envs", "CLWD", "Scripts", "pip")

    ## Install Pytorch with CUDA
    Command = f"{venv_path} install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128"
    os.system(Command)

    ## Install all other requirements
    requirements_path = os.path.join(current_path, "requirements.txt")
    Command = f"{venv_path} install -r {requirements_path}"
    os.system(Command)
else:
    print("Platform is not windows, please adjust the commands according to your platform.")