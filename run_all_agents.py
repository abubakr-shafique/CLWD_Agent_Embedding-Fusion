import os, sys


agent = "scripts/run_fusion_agent.py"
eval_agent = "scripts/evaluate_best_agent_model.py"
Folds = [0, 1, 2]

if sys.platform == "win32":

    for fold in Folds:
        print(f"Fold: {fold}\n")
        mode = "train"
        Command = f"python {agent} --fold {fold}"
        os.system(Command)

        mode = "eval"
        Command = f"python {eval_agent} --fold {fold}"
        os.system(Command)