import os
import torch
import torchvision
from torchvision import datasets, transforms
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from huggingface_hub import login
hf_token = "" ## Enter Your HF Access Token


def get_model(Network='UNI2-h', patch_size=224):

    if Network == 'Virchow2':

        from timm.layers import SwiGLUPacked

        login(token = hf_token)
        model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)

        data_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((patch_size, patch_size), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        print(f"{Network} Network Loaded")
    
    elif Network == 'UNI2-h':

        login(token = hf_token)
        timm_kwargs = {
            'img_size': 224, 
            'patch_size': 14, 
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5, 
            'embed_dim': 1536,
            'mlp_ratio': 2.66667*2,
            'num_classes': 0, 
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked, 
            'act_layer': torch.nn.SiLU, 
            'reg_tokens': 8, 
            'dynamic_img_size': True
        }
        model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)

        data_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((patch_size, patch_size), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        print(f"{Network} Network Loaded")

    elif Network == 'H-OPTIMUS-1':
    
            login(token = hf_token)
            model = timm.create_model("hf-hub:bioptimus/H-optimus-1", pretrained=True, init_values=1e-5, dynamic_img_size=False)
    
            data_transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((patch_size, patch_size), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                torchvision.transforms.ToTensor(),
                transforms.Normalize([0.707223, 0.578729, 0.703617], [0.211883, 0.230117, 0.177517])])
            print(f"{Network} Network Loaded")
    
    elif Network == 'Prism2':

        from transformers import AutoModel, AutoProcessor
        
        login(token = hf_token)
        model = AutoModel.from_pretrained("paige-ai/Prism2", trust_remote_code=True, torch_dtype="auto")
        data_transform = AutoProcessor.from_pretrained("paige-ai/Prism2", trust_remote_code=True)


    return model, data_transform