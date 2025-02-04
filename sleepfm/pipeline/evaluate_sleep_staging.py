import click
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
from loguru import logger
import wandb
import yaml
import os
from datetime import datetime
import sys
sys.path.append("../")
from utils import *

from models.models import SleepEventLSTMClassifier
from models.dataset import SleepEventClassificationDataset as Dataset
from models.dataset import sleep_event_finetune_full_collate_fn as collate_fn 
from tqdm import tqdm


@click.command("evaluate_sleep_staging")
@click.option("--config_path", type=str, default='../configs/config_finetune_sleep_events.yaml')
@click.option("--channel_groups_path", type=str, default='../configs/channel_groups.json' )
@click.option("--output_path", type=str, required=True)
@click.option("--split", type=str, default="test")
@click.option("--dataset", type=str, default=None)
def evaluate_sleep_staging(config_path, channel_groups_path, output_path, split, dataset):
    # Load configuration
    config = load_config(config_path)
    channel_groups = load_config(channel_groups_path)

    if "config.json" in os.listdir(output_path):
        config = load_data(os.path.join(output_path, "config.json"))

    # Set up logging
    logger.add("logs/evaluation_{time}.log", rotation="10 MB")
    logger.info("Loaded configuration file")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model_params = config['model_params']
    model_class = getattr(sys.modules[__name__], config['model'])
    logger.info(f"Model Class: {config['model']}")
    model = model_class(**model_params).to(device)
    model_name = type(model).__name__

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        logger.info(f"Using {torch.cuda.device_count()} GPUs")
    
    logger.info(f"Model initialized: {model_name}")
    total_layers, total_params = count_parameters(model)
    logger.info(f'Trainable parameters: {total_params / 1e6:.2f} million')
    logger.info(f'Number of layers: {total_layers}')

    logger.info(f'Loading Data...')

    # Load checkpoint
    checkpoint_path = os.path.join(output_path, "best.pth")
    if os.path.isfile(checkpoint_path):
        logger.info(f"Loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint)
        logger.info("Checkpoint loaded successfully")
    else:
        logger.error(f"Checkpoint '{checkpoint_path}' not found")
        return

    if dataset:
        config["dataset"] = dataset

    dataset_prefix = "_".join(config["dataset"].split(","))

    save_path = os.path.join(output_path, dataset_prefix, split)
    os.makedirs(save_path, exist_ok=True)

    # Initialize dataset and dataloaders
    test_dataset = Dataset(config, channel_groups, split=split)

    num_workers = config.get('num_workers', 4) 
    num_workers = 4
    logger.info(f'Number of workers: {num_workers}')

    batch_size = config.get('batch_size', 1)
    batch_size = 4
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    logger.info(f'Data Loaded!')

    # Validation loop at the end of each epoch
    model.eval()
    val_loss = 0.0
    all_targets = []
    all_logits = []
    all_outputs = []
    all_masks = []
    all_paths = []

    count = 0
    with torch.no_grad():
        for (x_data, y_data, padded_matrix, hdf5_path_list) in tqdm(test_loader, desc="Evaluating"):
            x_data, y_data, padded_matrix, hdf5_path_list = x_data.to(device), y_data.to(device), padded_matrix.to(device), list(hdf5_path_list)
            outputs, mask = model(x_data, padded_matrix)
            all_targets.append(y_data.cpu().numpy())
            all_outputs.append(torch.softmax(outputs, dim=-1).cpu().numpy())
            all_logits.append(outputs.cpu().numpy())
            all_masks.append(mask.cpu().numpy())
            all_paths.append(hdf5_path_list)


    targets_path = os.path.join(save_path, "all_targets.pickle")
    outputs_path = os.path.join(save_path, "all_outputs.pickle")
    logits_path = os.path.join(save_path, "all_logits.pickle")
    mask_path = os.path.join(save_path, "all_masks.pickle")
    file_paths = os.path.join(save_path, "all_paths.pickle")

    save_data(all_targets, targets_path)
    save_data(all_outputs, outputs_path)
    save_data(all_logits, logits_path)
    save_data(all_masks, mask_path)
    save_data(all_paths, file_paths)

    logger.info(f"All outputs saved at {outputs_path}")
    logger.info(f"Logits saved at {logits_path}")


if __name__ == "__main__":
    evaluate_sleep_staging()