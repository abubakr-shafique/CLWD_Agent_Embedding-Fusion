import os, sys
import argparse
import time
import pandas as pd
from tqdm import tqdm
import numpy as np
import random
import openslide
import torch
from pathlib import Path
import cv2
from PIL import Image

import project_dirs as pdir
import utils.tissue_segmentation as ts
import utils.wsi_patching as patching
import FMs.load_models as LM
import config.patch_embed_gen_config as patch_config

logg = False

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser(description='CLWD Patch Embedding Extraction')
parser.add_argument('--data_dir', type=str, default=patch_config.data_dir)
parser.add_argument('--embedding_dir', type=str, default=patch_config.embedding_dir)
parser.add_argument('--csv_dir', type=str, default=patch_config.csv_dir)
parser.add_argument('--model_name', type=str, default=patch_config.model_name, choices=['H-OPTIMUS-1', 'UNI2-h', 'Virchow2'])
parser.add_argument('--batch_size', type=int, default=patch_config.batch_size)
parser.add_argument('--seed', type=int, default=8, help="Random seed for reproducible experiment (default: 8)")
parser.add_argument('--target_patch_size', type=int, default=patch_config.target_patch_size)
parser.add_argument('--patch_overlap', type=float, default=patch_config.patch_overlap, help="1.0 Means no overlap, while 0.0 means 100% (full) overlap")
parser.add_argument('--tissue_area', type=int, default=patch_config.tissue_area, help="This means each patch should have atleast 80% tissue.")
parser.add_argument('--target_mag', type=int, default=patch_config.target_mag)
parser.add_argument('--smaller_level', type=int, default=patch_config.smaller_level)


args = parser.parse_args()


model, data_transform = LM.get_model(Network=args.model_name, patch_size=args.target_patch_size)
model = model.to(device)
model.eval()

def get_embeddings(images, model_name=args.model_name, batch_size=args.batch_size):
    """
    Generate embeddings for a list of PIL images while preserving
    the original image order.

    Parameters
    ----------
    images : list[PIL.Image.Image]
        List of PIL images to process.
        Transform applied to each PIL image.
    model_name : string, name of the model being used for inference.
    batch_size : int, default=32

    Returns
    -------
    torch.Tensor
        Embeddings in the same order as the input images.
        Shape: (N, embedding_dim)
    """

    all_embeddings = []

    with torch.no_grad():
        for start_idx in range(0, len(images), batch_size):

            # Get batch of PIL images
            batch_images = images[start_idx:start_idx + batch_size]

            # Apply transform
            batch = torch.stack([data_transform(img) for img in batch_images]).to(device)

            # Model inference
            embeddings = model(batch)

            # Move embeddings to CPU
            embeddings = embeddings.detach().cpu()
            if model_name == "Virchow2":
                cls_token = embeddings[:, 0]
                all_embeddings.append(cls_token)
            else:
                all_embeddings.append(embeddings)

    # Concatenate batches in the same order
    all_embeddings = torch.cat(all_embeddings, dim=0)

    return all_embeddings

if __name__ == '__main__':
    print("CLWD Generate Patch Embeddings.\n")
    set_seed(args.seed) # Set seed for reproducibility

    ## Get WSI_names by using CSV File
    # csv_data = pd.read_csv(args.csv_dir)
    # WSI_names = csv_data["WSI_ID"].to_list()

    ## Get WSI_names by using Data Directory and File Extension
    extensions = ['svs'] ## File extensions you want to search in the directory
    WSI_names = [fn for fn in os.listdir(args.data_dir) if any(fn.endswith(ext) for ext in extensions)] ##Gets all the files in the directory with the above mentioned extension

    embedding_gen_times = []
    WSI_read_times = []
    WSI_patches = []

    progressbar = tqdm(WSI_names)
    for i, WSI in enumerate(progressbar):
        path_obj = Path(WSI)
        if path_obj.suffix:
            WSI = path_obj.stem

        if logg:
            print(WSI)
        progressbar.set_description(f"Processing {WSI}")

        WSI_dir = os.path.join(args.data_dir, f"{WSI}.svs")

        thumb_Img_path = os.path.join(pdir.THUMBNAIL_DIR, f"{WSI}.jpg")
        thumb_Seg_path = os.path.join(pdir.THUMBNAIL_DIR, f"{WSI}-mask.png")

        if os.path.exists(WSI_dir): ## Check if SVS WSI exists in the Data Dir

            ## Open the slide Object
            slide = openslide.OpenSlide(WSI_dir)

            ### Perform Tissue Segmentation at thumbnail level
            if os.path.exists(thumb_Img_path) and os.path.exists(thumb_Seg_path): ## Check if Segmentation is already performed
                if logg:
                    print(f"{WSI}.svs tissue segmentation already done")
                thumb_Img = np.array(Image.open(thumb_Img_path).convert('RGB'))
                thumb_Seg = np.array(Image.open(thumb_Seg_path).convert('L'))
            else:

                thumb_Img, thumb_Seg = ts.Otsu_Seg(slide)
                
                cv2.imwrite(thumb_Img_path, thumb_Img)
                cv2.imwrite(thumb_Seg_path, thumb_Seg)

            ### Patch the WSI at thumbnail level
            obj_power = int(slide.properties['openslide.objective-power'])
            downsample_scale = int(obj_power/args.target_mag)
            patch_size_level0 = int(args.target_patch_size * downsample_scale)
            Level_0_dim = slide.level_dimensions[0]
            smaller_dim = slide.level_dimensions[args.smaller_level]

            scale_ratio = int(Level_0_dim[0]/smaller_dim[0])
            Patch_at_lowerMag = int(patch_size_level0/ scale_ratio)

            ### Resize the Mask and Tissue Thumbnail to Smaller Level such as args.smaller_level = 6
            thumb_Img = slide.read_region((0,0), args.smaller_level, smaller_dim)
            thumb_Seg = Image.fromarray(thumb_Seg)
            thumb_Seg = thumb_Seg.resize(smaller_dim)

            if logg:
                print(f"Scanner Magnification: {obj_power}")
                print(f"Level 0 Dimensions: {Level_0_dim}")
                print(f"Target Magnification: {args.target_mag}")
                print(f"Downsample Scale: {downsample_scale}")
                print(f"Target Patch Size: {args.target_patch_size }; at Level 0: {patch_size_level0}")

                print(f"Level {args.smaller_level} Dimensions: {smaller_dim}")
                print(f"Patch Size at Level {args.smaller_level}: {Patch_at_lowerMag}")
                print(f"Scale Ratio from Level to Level {args.smaller_level}: {scale_ratio}")

            ##Generate Patches on Lower Resolution
            patched_thumb_Img_path = os.path.join(pdir.VISUALIZATION_DIR, f"{WSI}.jpg")
            ### Patching Visualization
            if os.path.exists(patched_thumb_Img_path):
                patched_thumb_Img = Image.open(patched_thumb_Img_path).convert('RGB')
                patch_coords_level0 = np.load(os.path.join(pdir.PATCH_COORDS_DIR, f"{WSI}.npy"))
            else:
                patch_coords_smaller_level = patching.WSI_Patching(thumb_Seg, patch_size=Patch_at_lowerMag, iou_threshold=args.patch_overlap, Tissue_area=args.tissue_area)

                patched_thumb_Img = patching.Visualize_Patches(thumb_Img, patch_coords_smaller_level, Patch_at_lowerMag)
                patched_thumb_Img = Image.fromarray(patched_thumb_Img).convert('RGB')
                patched_thumb_Img.save(patched_thumb_Img_path)

                patch_coords_level0 = patching.patch_coord_scale(patch_coords_smaller_level, scale_ratio)
                np.save(os.path.join(pdir.PATCH_COORDS_DIR, f"{WSI}.npy"), patch_coords_level0)

            if logg:
                print(f"Total Patches: {len(patch_coords_level0)}")

            WSI_patches.append(len(patch_coords_level0))
            


            ### Generate Embedding Dir For the Inference Model
            embedding_save_dir = os.path.join(pdir.EMBEDDINGS_DIR, "patch", f"{args.model_name}_{args.target_patch_size}_{args.target_mag}x")
            os.makedirs(embedding_save_dir, exist_ok=True)

            WSI_embeddings_path = os.path.join(embedding_save_dir, f"{WSI}.pt")
            if os.path.exists(WSI_embeddings_path):
                WSI_embeddings = torch.load(WSI_embeddings_path, map_location=torch.device("cpu"))
                if logg:
                    print(f"{WSI} Embeddings are already generated using {args.model_name} model.")
            else:
                WSI_start_time = time.perf_counter()
                All_Images = []
                # os.makedirs(os.path.join(pdir.PATCH_IMAGES_DIR, WSI), exist_ok=True)
                ### Read Patches from the target level
                for patch_id, patch_coord in enumerate(patch_coords_level0):
                    x, y = patch_coord[0], patch_coord[1]
                    #### get the image from layer 2 (faster than layer 0)
                    Img = slide.read_region((y, x), 2, (args.target_patch_size, args.target_patch_size)).convert("RGB") ## This is for WSI Level 2 = 20x

                    #### get the image from layer 0 (take longer time)
                    # Img = slide.read_region((y, x), 0, (patch_size_level0, patch_size_level0)).convert("RGB") ## This is for WSI Level 2 = 20x
                    # Img = Img.resize((args.target_patch_size, args.target_patch_size)) ## if we want to save 20x

                    All_Images.append(Img)

                    ##### if want to save the image patches locally
                    # filepath = os.path.join(pdir.PATCH_IMAGES_DIR, WSI, f"{WSI}_{patch_id}.jpg")
                    # Img.save(filepath)
                WSI_elapsed_time = time.perf_counter() - WSI_start_time
                WSI_read_times.append(WSI_elapsed_time)

                embed_start_time = time.perf_counter()
                WSI_embeddings = get_embeddings(All_Images)
                embed_elapsed_time = time.perf_counter() - embed_start_time
                embedding_gen_times.append(embed_elapsed_time)

                if logg:
                    print(WSI_embeddings.shape)
                    print(f"WSI read time: {WSI_elapsed_time:.4f} seconds")
                    print(f"Wall-clock time: {embed_elapsed_time:.4f} seconds")

                torch.save(WSI_embeddings, WSI_embeddings_path)
                del All_Images

            ## clear memory
            slide.close()

        else:
            print(f"{WSI}.svs missing")

    ### Write the log File
    Log_file_path = os.path.join(pdir.ARTIFACTS_DIR, f"{args.model_name}_patch_embedding_generation_log.txt")
    file = open(Log_file_path, "a")
    my_log = (f"Model Name: {args.model_name},\n"
        f"Target Patch Size: {args.target_patch_size},\n"
        f"Target Magnification: {args.target_mag},\n"
        f"Scanner Magnification: {obj_power},\n"
        f"Target Patch Size {args.target_patch_size} at {obj_power}x Magnification: {patch_size_level0},\n"
        f"Downsample Scale to target Magnification: {downsample_scale},\n"
        f"Batch_Size: {args.batch_size},\n"
        f"Average WSI Processing time: {np.mean(embedding_gen_times)} seconds,\n"
        f"Average WSI Read time: {np.mean(WSI_read_times)} seconds,\n"
        f"Average Number of Patches per WSI: {int(np.mean(WSI_patches))},\n"
        f"Embedding Dimension: {WSI_embeddings.shape[1]}\n")
    file.write(my_log)
    file.close()




