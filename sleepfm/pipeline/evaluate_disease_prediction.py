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

from models.dataset import DiagnosisFinetuneFullCOXPHDataset as Dataset
from models.dataset import diagnosis_finetune_full_coxph_collate_fn as collate_fn 
from models.models import DiagnosisFinetuneFullLSTMCOXPH, DiagnosisFinetuneFullLSTMCOXPHWithDemo, DiagnosisFinetuneDemoOnlyEmbed
from tqdm import tqdm


@click.command("evaluate_diagnosis_coxph")
@click.option("--config_path", type=str, default='../configs/config_finetune_disease_prediction.yaml')
@click.option("--channel_groups_path", type=str, default='../configs/channel_groups.json' )
@click.option("--output_path", type=str, required=True)
@click.option("--split", type=str, default="test")
@click.option("--dataset", type=str, default=None)
def evaluate_diagnosis_coxph(config_path, channel_groups_path, output_path, split, dataset):
    # Load configuration
    config = load_config(config_path)
    channel_groups = load_data(channel_groups_path)

    if "config.json" in os.listdir(output_path):
        config = load_data(os.path.join(output_path, "config.json"))

    config["model_params"]["dropout"] = 0.0
    config["batch_size"] = 4

    # Set up logging
    logger.add("logs/evaluation_{time}.log", rotation="10 MB")
    logger.info("Loaded configuration file")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Initialize model
    model_params = config['model_params']
    model_class = getattr(sys.modules[__name__], config['model'])
    logger.info(f"Model Class: {config['model']}")
    if config['model'] in ["DiagnosisFinetuneDemoOnlyEmbed"]:
        embed_dim = config["model_params"]["embed_dim"]
        num_classes = config["model_params"]["num_classes"]
        model = model_class(embed_dim=embed_dim, num_classes=num_classes).to(device)
    else:
        model = model_class(**model_params).to(device)
    model_name = type(model).__name__

    sleep_stages = config.get("sleep_stages", "")
    if sleep_stages:
        sleep_stages_prefix = "_".join(sleep_stages)
    else:
        sleep_stages_prefix = ""

    if model_name in ["DiagnosisFinetuneFullLSTMCOXPHWithDemo"]:
        from models.dataset import DiagnosisFinetuneFullCOXPHWithDemoDataset as Dataset
        from models.dataset import diagnosis_finetune_full_coxph_with_demo_collate_fn as collate_fn 
    elif model_name in ["DiagnosisFinetuneFullLSTMCOXPH",]:
        from models.dataset import DiagnosisFinetuneFullCOXPHDataset as Dataset
        from models.dataset import diagnosis_finetune_full_coxph_collate_fn as collate_fn 
    elif model_name in ["DiagnosisFinetuneDemoOnlyEmbed"]:
        from models.dataset import DiagnosisFinetuneDemoOnlyDataset as Dataset
        from models.dataset import demo_only_collate_fn as collate_fn 


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
    # checkpoint_path = os.path.join(output_path, "checkpoint.pth")
    if os.path.isfile(checkpoint_path):
        logger.info(f"Loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path)
        # model.load_state_dict(checkpoint["model_state_dict"])
        model.load_state_dict(checkpoint)
        logger.info("Checkpoint loaded successfully")
    else:
        logger.error(f"Checkpoint '{checkpoint_path}' not found")
        return

    save_path = os.path.join(output_path, split)
    os.makedirs(save_path, exist_ok=True)

    if dataset:
        config["dataset"] = dataset

    if split == "external_validation":
        config["labels_path"] = "/oak/stanford/groups/jamesz/rthapa84/psg_fm/10_14_28/diagnosis_data/shhs_phewas_external_validation_tte_prediction_10_18_24_num_labels_6"

    # Initialize dataset and dataloaders
    test_dataset = Dataset(config, channel_groups, split=split)

    num_workers = config.get('num_workers', 4)  # Default to 4 workers if not specified
    num_workers = 4
    logger.info(f'Number of workers: {num_workers}')

    # batch_size = max(config.get('batch_size', 1), torch.cuda.device_count())
    batch_size = config.get('batch_size', 1)
    # batch_size = 2
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    logger.info(f'Data Loaded!')

    # Validation loop at the end of each epoch
    model.eval()
    val_loss = 0.0
    all_event_times = []
    all_is_event = []
    all_outputs = []
    all_paths = []

    with torch.no_grad():
        for item in tqdm(test_loader, desc="Evaluating"):
            if model_name in ["DiagnosisFinetuneFullLSTMCOXPHWithDemo""]:
                x_data, event_times, is_event, demo_feats, padded_matrix, hdf5_path_list = item
                x_data, event_times, is_event, demo_feats, padded_matrix, hdf5_path_list = x_data.to(device), event_times.to(device), is_event.to(device), demo_feats.to(device), padded_matrix.to(device), list(hdf5_path_list)
                outputs = model(x_data, padded_matrix, demo_feats)
            elif model_name in ["DiagnosisFinetuneDemoOnlyEmbed"]:
                demo_feats, event_times, is_event, hdf5_path_list = item
                demo_feats, event_times, is_event, hdf5_path_list = demo_feats.to(device), event_times.to(device), is_event.to(device), list(hdf5_path_list)
                outputs = model(demo_feats)
            else:
                x_data, event_times, is_event, padded_matrix, hdf5_path_list = item
                x_data, event_times, is_event, padded_matrix, hdf5_path_list = x_data.to(device), event_times.to(device), is_event.to(device), padded_matrix.to(device), list(hdf5_path_list)
                outputs = model(x_data, padded_matrix)
            
            logits = outputs.cpu().numpy()
            all_outputs.append(logits)
            all_event_times.append(event_times.cpu().numpy())
            all_is_event.append(is_event.cpu().numpy())
            all_paths.append(hdf5_path_list)

    all_outputs = np.concatenate(all_outputs, axis=0)
    all_event_times = np.concatenate(all_event_times, axis=0)
    all_is_event = np.concatenate(all_is_event, axis=0)
    all_paths = np.concatenate(all_paths)

    outputs_path = os.path.join(save_path, "all_outputs.pickle")
    event_times_path = os.path.join(save_path, "all_event_times.pickle")
    is_event_path = os.path.join(save_path, "all_is_event.pickle")
    file_paths = os.path.join(save_path, "all_paths.pickle")

    save_data(all_outputs, outputs_path)
    save_data(all_event_times, event_times_path)
    save_data(all_is_event, is_event_path)
    save_data(all_paths, file_paths)

    logger.info(f"All outputs saved at {outputs_path}")


if __name__ == "__main__":
    evaluate_diagnosis_coxph()
