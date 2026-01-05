import click
import torch
import torch.nn as nn
import torch.optim as optim
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
from models.models import DiagnosisFinetuneFullLSTMCOXPHWithDemo, DiagnosisFinetuneDemoOnlyEmbed, DiagnosisFullSupervisedLSTMCOXPHWithDemoEmbed
from tqdm import tqdm
import pandas as pd
from torch.cuda.amp import autocast, GradScaler
import random

def set_seed(seed=10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cox_ph_loss(hazards, event_times, is_event):
    # Sort event times and get corresponding indices for sorting other tensors
    event_times, sorted_idx = event_times.sort(dim=0, descending=True)
    hazards = hazards.gather(0, sorted_idx)
    is_event = is_event.gather(0, sorted_idx)

    log_cumulative_hazard = torch.logcumsumexp(hazards.float(), dim=0)

    # Calculate losses for all labels simultaneously
    losses = (hazards - log_cumulative_hazard) * is_event
    losses = -losses  # Negative for maximization

    # Average loss per label
    label_loss = losses.sum(dim=0) / (is_event.sum(dim=0) + 1e-9)  # Avoid division by zero

    # Average across labels
    total_loss = label_loss.mean()

    return total_loss


@click.command("finetune_diagnosis")
@click.option("--config_path", type=str, default='../configs/config_finetune_diagnosis_coxph.yaml')
@click.option("--channel_groups_path", type=str, default='../configs/channel_groups.json')
@click.option("--checkpoint_path", type=str, default=None)
@click.option("--split_path", type=str, default=None)
def finetune_diagnosis(config_path, channel_groups_path, checkpoint_path, split_path):
    # Load configuration
    config = load_config(config_path)
    channel_groups = load_data(channel_groups_path)

    labels_path = config["labels_path"]
    is_event_df = pd.read_csv(os.path.join(labels_path, "is_event.csv"))
    num_classes = is_event_df.shape[1] - 1
    config["model_params"]["num_classes"] = num_classes

    set_seed(config["seed"])

    prefix = labels_path.split("/")[-1]

    if "demo" in config["model"].lower():
        label_file_name = os.path.basename(config["demo_labels_path"]).split(".")[0]
    else:
        label_file_name = ""

    # Model saving path
    current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if split_path:
        config["split_path"] = split_path
    
    split_path = config["split_path"]
    split_prefix = split_path.split("/")[-1].split(".")[0]
    channel_like = config["modality_types"]
    channel_like_string = "_".join(channel_like)

    sleep_stages = config.get("sleep_stages", "")
    if sleep_stages:
        sleep_stages_prefix = "_".join(sleep_stages)
    else:
        sleep_stages_prefix = ""

    epochs = config["epochs"]
    batch_size = config["batch_size"]

    if checkpoint_path:
        output = checkpoint_path
        config = load_data(os.path.join(output, "config.json"))
    else:
        output = os.path.join(config["model_path"], f"{config['model']}_{config['dataset']}_{prefix}_{label_file_name}_{channel_like_string}_{sleep_stages_prefix}_ep_{epochs}_bs_{batch_size}/{split_prefix}")
        os.makedirs(output, exist_ok=True)
    
    # Set up logging
    logger.info(f"Random seed: {config['seed']}")
    logger.info(f"Model path: {output}")
    logger.info(f"Split Path: {config['split_path']}")
    logger.add("logs/training_{time}.log", rotation="10 MB")
    logger.info("Loaded configuration file")
    logger.info(f"Batch Size {config['batch_size']}")
    logger.info(f"Processing for sleep stage {sleep_stages}")

    # config['batch_size'] = 4
    # config['use_wandb'] = False

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

    if model_name in ["DiagnosisFinetuneFullLSTMCOXPHWithDemo"]:
        from models.dataset import DiagnosisFinetuneFullCOXPHWithDemoDataset as Dataset
        from models.dataset import diagnosis_finetune_full_coxph_with_demo_collate_fn as collate_fn 
    elif model_name in ["DiagnosisFinetuneDemoOnlyEmbed"]:
        from models.dataset import DiagnosisFinetuneDemoOnlyDataset as Dataset
        from models.dataset import demo_only_collate_fn as collate_fn 
    elif config["model"] in ["DiagnosisFullSupervisedLSTMCOXPHWithDemoEmbed"]:
        from models.dataset import SupervisedDiagnosisFullCOXPHWithDemoDataset as Dataset
        from models.dataset import supervised_diagnosis_full_coxph_with_demo_collate_fn as collate_fn

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        logger.info(f"Using {torch.cuda.device_count()} GPUs")

    logger.info(f"Model initialized: {model_name}")
    total_layers, total_params = count_parameters(model)
    logger.info(f'Trainable parameters: {total_params / 1e6:.2f} million')
    logger.info(f'Number of layers: {total_layers}')

    logger.info(f'Loading Data...')

    # Initialize dataset and dataloaders
    train_dataset = Dataset(config, channel_groups, split="train")
    val_dataset = Dataset(config, channel_groups, split="validation")

    num_workers = config.get('num_workers', 4)  # Default to 4 workers if not specified
    logger.info(f'Number of workers: {num_workers}')

    # batch_size = max(config.get('batch_size', 1), torch.cuda.device_count())
    batch_size = config.get('batch_size', 1)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)

    logger.info(f'Data Loaded!')

    # Optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    # Initialize GradScaler for mixed precision training
    # scaler = GradScaler()

    start_epoch = 0
    best_val_loss = float('inf')
    if checkpoint_path:
        checkpoint_path = os.path.join(output, "checkpoint.pth")
        if os.path.isfile(checkpoint_path):
            logger.info(f"Loading checkpoint '{checkpoint_path}'")
            checkpoint = torch.load(checkpoint_path)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"Checkpoint loaded. Resuming from epoch {start_epoch}.")
        else:
            logger.info(f"Initializing the model from scratch...")

    # Set up Weights & Biases
    if config["use_wandb"]:
        current_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        wandb.init(project="PSG-fm", name=f"run_at_{current_timestamp}", config=config)
    
    # Training loop
    num_epochs = config.get('epochs', 8)
    accumulation_steps = config.get('accumulation_steps', 8)
    save_iter = config.get('save_iter', 100)
    log_interval = config.get('log_interval', 10)
    eval_iter = config.get('eval_iter', 50)

    best_val_loss = float('inf')
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        for i, item in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            if model_name in ["DiagnosisFinetuneFullLSTMCOXPHWithDemo", "DiagnosisFullSupervisedLSTMCOXPHWithDemoEmbed"]:
                x_data, event_times, is_event, demo_feats, padded_matrix, hdf5_path_list = item
                x_data, event_times, is_event, demo_feats, padded_matrix, hdf5_path_list = x_data.to(device), event_times.to(device), is_event.to(device), demo_feats.to(device), padded_matrix.to(device), list(hdf5_path_list)
                hazards = model(x_data, padded_matrix, demo_feats)
            elif model_name in ["DiagnosisFinetuneDemoOnlyEmbed"]:
                demo_feats, event_times, is_event, hdf5_path_list = item
                demo_feats, event_times, is_event, hdf5_path_list = demo_feats.to(device), event_times.to(device), is_event.to(device), list(hdf5_path_list)
                hazards = model(demo_feats)
            else:
                x_data, event_times, is_event, padded_matrix, hdf5_path_list = item
                x_data, event_times, is_event, padded_matrix, hdf5_path_list = x_data.to(device), event_times.to(device), is_event.to(device), padded_matrix.to(device), list(hdf5_path_list)
                hazards = model(x_data, padded_matrix)
            loss = cox_ph_loss(hazards, event_times, is_event)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            running_loss += loss.item()

            if (i + 1) % log_interval == 0:
                avg_loss = running_loss / (i + 1)
                logger.info(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {avg_loss:.4f}")
                if config["use_wandb"]:
                    wandb.log({"Train Loss": avg_loss, "Step": (epoch * len(train_loader)) + i + 1})

            if (i + 1) % save_iter == 0:
                checkpoint_path = os.path.join(output, f"checkpoint.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss
                }, checkpoint_path)
                save_data(config, os.path.join(output, "config.json"))
                logger.info(f"Checkpoint saved at {checkpoint_path}")

            if (i + 1) % eval_iter == 0:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for item in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{num_epochs}"):
                        if model_name in ["DiagnosisFinetuneFullLSTMCOXPHWithDemo", "DiagnosisFullSupervisedLSTMCOXPHWithDemoEmbed"]:
                            x_data, event_times, is_event, demo_feats, padded_matrix, hdf5_path_list = item
                            x_data, event_times, is_event, demo_feats, padded_matrix, hdf5_path_list = x_data.to(device), event_times.to(device), is_event.to(device), demo_feats.to(device), padded_matrix.to(device), list(hdf5_path_list)
                            hazards = model(x_data, padded_matrix, demo_feats)
                        elif model_name in ["DiagnosisFinetuneDemoOnlyEmbed"]:
                            demo_feats, event_times, is_event, hdf5_path_list = item
                            demo_feats, event_times, is_event, hdf5_path_list = demo_feats.to(device), event_times.to(device), is_event.to(device), list(hdf5_path_list)
                            hazards = model(demo_feats)
                        else:
                            x_data, event_times, is_event, padded_matrix, hdf5_path_list = item
                            x_data, event_times, is_event, padded_matrix, hdf5_path_list = x_data.to(device), event_times.to(device), is_event.to(device), padded_matrix.to(device), list(hdf5_path_list)
                            hazards = model(x_data, padded_matrix)
                        loss = cox_ph_loss(hazards, event_times, is_event)
                        val_loss += loss.item()

                val_loss /= len(val_loader)
                logger.info(f"Validation Loss after Epoch [{epoch + 1}/{num_epochs}], Iteration [{i + 1}]: {val_loss:.4f}")

                if config["use_wandb"]:
                    wandb.log({
                        "Validation Loss": val_loss,
                        "Step": (epoch * len(train_loader)) + i + 1
                    })

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_path = os.path.join(output, "best.pth")
                    torch.save(model.state_dict(), best_model_path)
                    save_data(config, os.path.join(output, "config.json"))
                    logger.info(f"Best model saved at {best_model_path}")

                model.train()

        # Log epoch loss
        epoch_loss = running_loss / len(train_loader)
        logger.info(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {epoch_loss:.4f}")

        if config["use_wandb"]:
            wandb.log({"Epoch": epoch + 1, "Loss": epoch_loss})
        
        # Validation loop at the end of each epoch
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for item in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{num_epochs}"):
                if model_name in ["DiagnosisFinetuneFullLSTMCOXPHWithDemo", "DiagnosisFullSupervisedLSTMCOXPHWithDemoEmbed"]:
                    x_data, event_times, is_event, demo_feats, padded_matrix, hdf5_path_list = item
                    x_data, event_times, is_event, demo_feats, padded_matrix, hdf5_path_list = x_data.to(device), event_times.to(device), is_event.to(device), demo_feats.to(device), padded_matrix.to(device), list(hdf5_path_list)
                    hazards = model(x_data, padded_matrix, demo_feats)
                elif model_name in ["DiagnosisFinetuneDemoOnlyEmbed"]:
                    demo_feats, event_times, is_event, hdf5_path_list = item
                    demo_feats, event_times, is_event, hdf5_path_list = demo_feats.to(device), event_times.to(device), is_event.to(device), list(hdf5_path_list)
                    hazards = model(demo_feats)
                else:
                    x_data, event_times, is_event, padded_matrix, hdf5_path_list = item
                    x_data, event_times, is_event, padded_matrix, hdf5_path_list = x_data.to(device), event_times.to(device), is_event.to(device), padded_matrix.to(device), list(hdf5_path_list)
                    hazards = model(x_data, padded_matrix)
                loss = cox_ph_loss(hazards, event_times, is_event)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        logger.info(f"Validation Loss after Epoch [{epoch + 1}/{num_epochs}]: {val_loss:.4f}")

        if config["use_wandb"]:
            wandb.log({
                "Validation Loss": val_loss,
                "Epoch": epoch + 1
            })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(output, "best.pth")
            torch.save(model.state_dict(), best_model_path)
            save_data(config, os.path.join(output, "config.json"))
            logger.info(f"Best model saved at {best_model_path}")

        scheduler.step(val_loss)
        model.train()

if __name__ == "__main__":
    finetune_diagnosis()