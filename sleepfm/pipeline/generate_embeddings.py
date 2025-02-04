import yaml
import torch
from torch import nn
from loguru import logger
import os
import sys
sys.path.append("../")
from utils import *
from models.dataset import SetTransformerDataset, collate_fn
from models.models import SetTransformer
import click
import time
import math
import datetime
import numpy as np
import tqdm
import shutil
import wandb
import h5py


@click.command("generate_embeddings")
@click.option("--model_path", type=str, default='path')
@click.option("--dataset_name", type=str, default='mesa')
@click.option("--channel_groups_path", type=str, default='../configs/channel_groups.json')
@click.option("--split_path", type=str, default='../configs/dataset_split.json')
@click.option("--splits", type=str, default='train,validation,test')
@click.option("--num_workers", type=int, default=16)
@click.option("--batch_size", type=int, default=128)
def generate_embeddings(
    model_path,
    dataset_name, 
    channel_groups_path, 
    split_path,
    splits,
    num_workers, 
    batch_size
):
    config_path = os.path.join(model_path, "config.json")
    config = load_config(config_path)
    channel_groups = load_data(channel_groups_path)

    current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    dataset_name = dataset_name.lower()

    output = os.path.join(model_path, f"{dataset_name}")
    output_5min_agg = os.path.join(model_path, f"{dataset_name}_5min_agg")
    os.makedirs(output, exist_ok=True)
    os.makedirs(output_5min_agg, exist_ok=True)

    modality_types = config["modality_types"]
    in_channels = config["in_channels"]
    patch_size = config["patch_size"]
    embed_dim = config["embed_dim"]
    num_heads = config["num_heads"]
    num_layers = config["num_layers"]
    pooling_head = config["pooling_head"]
    dropout = 0.0

    data_path = config["data_path"]

    logger.info(f"Output Path: {output}")
    logger.info(f"Output 5 Min Agg Path: {output_5min_agg}")
    logger.info(f"modality_types: {modality_types}")

    logger.info(f"Batch Size: {batch_size}; Number of Workers: {num_workers}")

    device = torch.device("cuda")
    logger.info(f"Device set to Cuda")

    start = time.time()
    split_dataset = load_data(split_path)
    splits = splits.split(",")

    if dataset_name.lower() in ["shhs1", "shhs2"]:
        path_to_data = os.path.join(data_path, f"SHHS/{dataset_name}")
        hdf5_paths = [os.path.join(path_to_data, file_name) for file_name in os.listdir(path_to_data)]
    else:
        hdf5_paths = []
        for split in splits:
            filtered_files = [fp for fp in split_dataset[split] if dataset_name in fp.lower()]
            hdf5_paths += filtered_files
        
        hdf5_paths = [os.path.join(data_path, file) for file in hdf5_paths]

    logger.info(f"Number of files to process: {len(hdf5_paths)}")

    dataset = SetTransformerDataset(config, channel_groups, hdf5_paths=hdf5_paths, split="test")
    dataloader = torch.utils.data.DataLoader(dataset, 
                                             batch_size=batch_size, 
                                             num_workers=num_workers, 
                                             shuffle=False, 
                                             collate_fn=collate_fn)

    logger.info(f"Dataset loaded in {time.time() - start:.1f} seconds")

    model_class = getattr(sys.modules[__name__], config['model'])
    logger.info(f"Model Class: {config['model']}")
    model = model_class(in_channels, patch_size, embed_dim, num_heads, num_layers, pooling_head=pooling_head, dropout=dropout)
    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)
    total_layers, total_params = count_parameters(model)
    logger.info(f'Trainable parameters: {total_params / 1e6:.2f} million')
    logger.info(f'Number of layers: {total_layers}')

    checkpoint = torch.load(os.path.join(model_path, "best.pt"))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    with torch.no_grad():
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for batch in dataloader:
                batch_data, mask_list, file_paths, dset_names_list, chunk_starts = batch
                (bas, resp, ekg, emg) = batch_data
                (mask_bas, mask_resp, mask_ekg, mask_emg) = mask_list

                bas = bas.to(device, dtype=torch.float)
                resp = resp.to(device, dtype=torch.float)
                ekg = ekg.to(device, dtype=torch.float)
                emg = emg.to(device, dtype=torch.float)

                mask_bas = mask_bas.to(device, dtype=torch.bool)
                mask_resp = mask_resp.to(device, dtype=torch.bool)
                mask_ekg = mask_ekg.to(device, dtype=torch.bool)
                mask_emg = mask_emg.to(device, dtype=torch.bool)

                embeddings = [
                    model(bas, mask_bas),
                    model(resp, mask_resp),
                    model(ekg, mask_ekg),
                    model(emg, mask_emg),
                ]

                embeddings_new = [e[0].unsqueeze(1) for e in embeddings]

                for i in range(len(file_paths)):
                    file_path = file_paths[i]
                    chunk_start = chunk_starts[i]
                    subject_id = os.path.basename(file_path).split('.')[0]
                    output_path = os.path.join(output_5min_agg, f"{subject_id}.hdf5")

                    with h5py.File(output_path, 'a') as hdf5_file:
                        for modality_idx, modality_type in enumerate(config["modality_types"]):
                            if modality_type in hdf5_file:
                                dset = hdf5_file[modality_type]
                                chunk_start_correct = chunk_start // (embed_dim * 5 * 60)
                                chunk_end = chunk_start_correct + embeddings_new[modality_idx][i].shape[0]
                                if dset.shape[0] < chunk_end:
                                    dset.resize((chunk_end,) + embeddings_new[modality_idx][i].shape[1:])
                                dset[chunk_start_correct:chunk_end] = embeddings_new[modality_idx][i].cpu().numpy()
                            else:
                                hdf5_file.create_dataset(modality_type, data=embeddings_new[modality_idx][i].cpu().numpy(), chunks=(embed_dim,) + embeddings_new[modality_idx][i].shape[1:], maxshape=(None,) + embeddings_new[modality_idx][i].shape[1:])

                embeddings_new = [e[1] for e in embeddings]

                for i in range(len(file_paths)):
                    file_path = file_paths[i]
                    chunk_start = chunk_starts[i]
                    subject_id = os.path.basename(file_path).split('.')[0]
                    output_path = os.path.join(output, f"{subject_id}.hdf5")

                    with h5py.File(output_path, 'a') as hdf5_file:
                        for modality_idx, modality_type in enumerate(config["modality_types"]):
                            if modality_type in hdf5_file:
                                dset = hdf5_file[modality_type]
                                chunk_start_correct = chunk_start // (embed_dim * 5)
                                chunk_end = chunk_start_correct + embeddings_new[modality_idx][i].shape[0]
                                if dset.shape[0] < chunk_end:
                                    dset.resize((chunk_end,) + embeddings_new[modality_idx][i].shape[1:])
                                dset[chunk_start_correct:chunk_end] = embeddings_new[modality_idx][i].cpu().numpy()
                            else:
                                hdf5_file.create_dataset(modality_type, data=embeddings_new[modality_idx][i].cpu().numpy(), chunks=(embed_dim,) + embeddings_new[modality_idx][i].shape[1:], maxshape=(None,) + embeddings_new[modality_idx][i].shape[1:])
                pbar.update()


class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


if __name__ == '__main__':
    generate_embeddings()