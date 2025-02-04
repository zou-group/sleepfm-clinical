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
from models.models import SleepEventLSTMClassifier
from models.dataset import SleepEventClassificationDataset as Dataset
from models.dataset import sleep_event_finetune_full_collate_fn as collate_fn 
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F


def masked_cross_entropy_loss(outputs, y_data, mask):
    # Reshape outputs and labels to (B * seq_len, num_classes) and (B * seq_len,)
    B, seq_len, num_classes = outputs.shape
    outputs = outputs.reshape(B * seq_len, num_classes)
    y_data = y_data.reshape(B * seq_len).long()  # Convert y_data to Long for cross_entropy
    mask = mask.reshape(B * seq_len)

    class_weights = {0: 1,
                    1: 4,
                    2: 2,
                    3: 4,
                    4: 3
                    }

    weights_tensor = torch.ones(num_classes)
    for cls, weight in class_weights.items():
        weights_tensor[cls] = weight

    weights_tensor = torch.tensor(weights_tensor, device=outputs.device) 

    loss = F.cross_entropy(outputs, y_data, weight=weights_tensor, reduction='none')

    loss = loss * (mask == 0).float() 

    loss = loss.sum() / (mask == 0).float().sum()
    
    return loss


@click.command("finetune_sleep_staging")
@click.option("--config_path", type=str, default='../configs/config_finetune_sleep_events.yaml')
@click.option("--channel_groups_path", type=str, default='../configs/channel_groups.json' )
@click.option("--checkpoint_path", type=str, default=None)
@click.option("--split_path", type=str, default=None)
@click.option("--train_split", type=str, default="train")
def finetune_sleep_staging(config_path, channel_groups_path, checkpoint_path, split_path, train_split):
    # Load configuration
    config = load_config(config_path)
    channel_groups = load_config(channel_groups_path)

    prefix = config["labels_path"].split("/")[-1]
    current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if split_path:
        config["split_path"] = split_path
    
    split_path = config["split_path"]
    channel_like = config["channel_like"]
    channel_like_string = "_".join(channel_like)

    dataset_prefix = "_".join(config["dataset"].split(","))

    if checkpoint_path:
        output = checkpoint_path
        config = load_data(os.path.join(output, "config.json"))
    else:
        output = os.path.join(config["model_path"], f"{config['model']}_{dataset_prefix}_{prefix}_{channel_like_string}")
        os.makedirs(output, exist_ok=True)
    
    # Set up logging
    logger.info(f"Model path: {output}")
    logger.info(f"Split Path: {config['split_path']}")
    logger.add("logs/training_{time}.log", rotation="10 MB")
    logger.info("Loaded configuration file")
    logger.info(f"Batch Size {config['batch_size']}")


    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Initialize model
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

    # Initialize dataset and dataloaders
    train_dataset = Dataset(config, channel_groups, split=train_split)
    val_dataset = Dataset(config, channel_groups, split="validation")

    num_workers = config.get('num_workers', 4)  # Default to 4 workers if not specified
    logger.info(f'Number of workers: {num_workers}')

    batch_size = config.get('batch_size', 1)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    logger.info(f'Data Loaded!')

    # Optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

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
        for i, (x_data, y_data, padded_matrix, hdf5_path_list) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            
            x_data, y_data, padded_matrix, hdf5_path_list = x_data.to(device), y_data.to(device), padded_matrix.to(device), list(hdf5_path_list)
            outputs, mask = model(x_data, padded_matrix)

            loss = masked_cross_entropy_loss(outputs, y_data, mask)
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
                all_targets = []
                all_outputs = []
                with torch.no_grad():
                    for (x_data, y_data, padded_matrix, hdf5_path_list) in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{num_epochs}"):
                        x_data, y_data, padded_matrix, hdf5_path_list = x_data.to(device), y_data.to(device), padded_matrix.to(device), list(hdf5_path_list)
                        outputs, mask = model(x_data, padded_matrix)
                        loss = masked_cross_entropy_loss(outputs, y_data, mask)
                        val_loss += loss.item()
                        all_targets.append(y_data.cpu().numpy())
                        all_outputs.append(torch.sigmoid(outputs).cpu().numpy())

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
        all_targets = []
        all_outputs = []
        with torch.no_grad():
            for (x_data, y_data, padded_matrix, hdf5_path_list) in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{num_epochs}"):
                x_data, y_data, padded_matrix, hdf5_path_list = x_data.to(device), y_data.to(device), padded_matrix.to(device), list(hdf5_path_list)
                outputs, mask = model(x_data, padded_matrix)
                loss = masked_cross_entropy_loss(outputs, y_data, mask)
                val_loss += loss.item()
                all_targets.append(y_data.cpu().numpy())
                all_outputs.append(torch.sigmoid(outputs).cpu().numpy())

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
    finetune_sleep_staging()