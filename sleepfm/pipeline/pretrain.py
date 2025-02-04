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
from torch.optim import AdamW


def run_iter(batch, num_modalities, model, device, mode, temperature, batch_size, ij):
    batch_data, mask_list, *_ = batch
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

    # TODO: might be able to pack CNNs and set transformer to do a single call, rather than separate forward passes
    # Won't work if you have separate CNN params for each modality
    emb = [
        model(bas, mask_bas),
        model(resp, mask_resp),
        model(ekg, mask_ekg),
        model(emg, mask_emg),
    ]

    emb = [e[0] for e in emb]

    for i in range(num_modalities):
        emb[i] = torch.nn.functional.normalize(emb[i])

    if mode == "pairwise":
        loss = 0.
        pairwise_loss = np.zeros((num_modalities, num_modalities), dtype=float)
        correct = np.zeros((num_modalities, num_modalities), dtype=int)
        pairs = np.zeros((num_modalities, num_modalities), dtype=int)

        for i in range(num_modalities ):
            for j in range(i + 1, num_modalities):

                logits = torch.matmul(emb[i], emb[j].transpose(0, 1)) * torch.exp(temperature)
                labels = torch.arange(logits.shape[0], device=device)
    
                l = torch.nn.functional.cross_entropy(logits, labels, reduction="sum")
                loss += l
                pairwise_loss[i, j] = l.item()
                if len(logits) != 0:
                    correct[i, j] = (torch.argmax(logits, axis=0) == labels).sum().item()
                else:
                    correct[i, j] = 0
                pairs[i, j] = batch_size
                
                l = torch.nn.functional.cross_entropy(logits.transpose(0, 1), labels.to(device), reduction="sum")
                loss += l
                pairwise_loss[j, i] = l.item()
                if len(logits) != 0:
                    correct[j, i] = (torch.argmax(logits, axis=1) == labels).sum().item()
                else:
                    correct[j, i] = 0
                pairs[j, i] = batch_size
        loss /= len(ij)
    if mode == "leave_one_out":
        loss = 0.
        pairwise_loss = np.zeros((num_modalities, 2), dtype=float)
        correct = np.zeros((num_modalities, 2), dtype=int)
        pairs = np.zeros((num_modalities, 2), dtype=int)

        for i in range(num_modalities):
            other_emb = torch.stack([emb[j] for j in list(range(i)) + list(range(i + 1, num_modalities))]).sum(0) / (num_modalities - 1)
            logits = torch.matmul(emb[i], other_emb.transpose(0, 1)) * torch.exp(temperature)
            labels = torch.arange(logits.shape[0], device=device)
    
            l = torch.nn.functional.cross_entropy(logits, labels, reduction="sum")
            loss += l
            pairwise_loss[i, 0] = l.item()
            if len(logits) != 0:
                correct[i, 0] = (torch.argmax(logits, axis=0) == labels).sum().item()
            else:
                correct[i, 0] = 0
            pairs[i, 0] = batch_size
            
            l = torch.nn.functional.cross_entropy(logits.transpose(0, 1), labels.to(device), reduction="sum")
            loss += l
            pairwise_loss[i, 1] = l.item()
            if len(logits) != 0:
                correct[i, 1] = (torch.argmax(logits, axis=1) == labels).sum().item()
            else:
                correct[i, 1] = 0
            pairs[i, 1] = batch_size
        loss /= num_modalities * 2

    return loss, pairwise_loss, correct, pairs


@click.command("pretrain")
@click.option("--config_path", type=str, default='../configs/config_set_transformer_contrastive.yaml')
@click.option("--channel_groups_path", type=str, default='../configs/channel_groups.json' )
@click.option("--checkpoint_path", type=str, default=None)
@click.option("--use_wandb", type=str, default=None)
def pretrain(
    config_path, 
    channel_groups_path, 
    checkpoint_path, 
    use_wandb
):

    config = load_config(config_path)
    channel_groups = load_data(channel_groups_path)
    current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if checkpoint_path:
        output = checkpoint_path
        logger.info("Loading saved config")
        config_path = os.path.join(output, "config.json")
        config = load_config(config_path)
    else:
        output = os.path.join(config["save_path"], f"{config['model']}/{config['mode']}_{config['embed_dim']}_patch_size_{config['patch_size']}")
        os.makedirs(output, exist_ok=True)

    data_path = config["data_path"]

    modality_types = config["modality_types"]
    lr = config["lr"]
    lr_step_period = config["lr_step_period"]
    gamma = config["gamma"]
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    temperature = config["temperature"]
    momentum = config["momentum"]
    num_workers = config["num_workers"]
    weight_decay = config["weight_decay"]
    mode = config["mode"]
    in_channels = config["in_channels"]
    sampling_freq = config["sampling_freq"]
    patch_size = config["patch_size"]
    embed_dim = config["embed_dim"]
    num_heads = config["num_heads"]
    num_layers = config["num_layers"]
    pooling_head = config["pooling_head"]
    dropout = config["dropout"]
    log_interval = config["log_interval"]
    model_name = config["model"]
    mode = config["mode"]

    config["use_wandb"] = False

    if config["use_wandb"]:
        wandb.init(project="PSG-fm", name=f"run_at_{current_timestamp}", config=config)

    os.environ['WANDB_DIR'] = output

    temperature = torch.nn.parameter.Parameter(torch.as_tensor(temperature))

    logger.info(f"Output Path: {output}")
    logger.info(f"modality_types: {modality_types}")
    logger.info(f"Training Mode: {mode}")

    logger.info(f"Batch Size: {batch_size}; Number of Workers: {num_workers}")
    logger.info(f"Weight Decay: {weight_decay}; Learning Rate: {lr}; Learning Step Period: {lr_step_period}")

    device = torch.device("cuda")
    logger.info(f"Device set to Cuda")

    num_modalities = len(modality_types)
    ij = sum([((i, j), (j, i)) for i in range(len(modality_types)) for j in range(i + 1, len(modality_types))], ())

    start = time.time()
    # dataset_class = getattr(sys.modules[__name__], config['dataloader'])
    dataset = {
        split: SetTransformerDataset(config, channel_groups, split=split)
        for split in ["pretrain", "validation"]
    }

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

    optim_params = list(model.parameters())
    optim_params.append(temperature)  

    optim = torch.optim.SGD(
        optim_params,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )

    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=lr_step_period, gamma=gamma)

    epoch_resume = 0
    best_loss = math.inf

    if os.path.isfile(os.path.join(output, "checkpoint.pt")):
        checkpoint = torch.load(os.path.join(output, "checkpoint.pt"))
        model.load_state_dict(checkpoint[f"state_dict"])

        # Loading temperature and other checkpointed parameters
        with torch.no_grad():
            temperature.fill_(checkpoint["temperature"])
        optim.load_state_dict(checkpoint["optim_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_dict"])

        # Other checkpointed values
        epoch_resume = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]
        logger.info(f"Resuming from epoch {epoch_resume}\n")
    else:
        logger.info("Starting from scratch")
    os.makedirs(os.path.join(output, "log"), exist_ok=True)
    with open(os.path.join(output, "log", "{}.tsv".format(datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))), "w") as f:
        f.write("Epoch\tSplit\tTotal Loss\t")
        if mode == "pairwise":
            f.write("".join(f"{modality_types[i]}-{modality_types[j]} Loss\t" for (i, j) in ij))
            f.write("".join(f"{modality_types[i]}-{modality_types[j]} Accuracy\t" for (i, j) in ij))
        elif mode == "leave_one_out":
            f.write("".join(f"{modality_types[i]}-other Loss\tother-{modality_types[i]} Loss\t" for i in range(len(modality_types))))
            f.write("".join(f"{modality_types[i]}-other Accuracy\tother-{modality_types[i]} Accuracy\t" for i in range(len(modality_types))))
            
        f.write("Temperature\n")
        f.flush()
        
        count_iter = 1
        for epoch in range(epoch_resume, epochs):
            split = "pretrain"
            dataloader = torch.utils.data.DataLoader(dataset[split], batch_size=batch_size, num_workers=num_workers, shuffle=True, collate_fn=collate_fn, drop_last=(split == "pretrain"))
            model.train(split == "pretrain")
            if mode == "pairwise":
                total_loss = 0.
                total_pairwise_loss = np.zeros((num_modalities, num_modalities), dtype=float)
                total_correct = np.zeros((num_modalities, num_modalities), dtype=int)
                total_n = 0
                total_pairs = np.zeros((num_modalities, num_modalities), dtype=int)
            elif mode == "leave_one_out":
                total_loss = 0.
                total_pairwise_loss = np.zeros((num_modalities, 2), dtype=float)
                total_correct = np.zeros((num_modalities, 2), dtype=int)
                total_n = 0
                total_pairs = np.zeros((num_modalities, 2), dtype=int)

            with torch.set_grad_enabled(split == "pretrain"):
                with tqdm.tqdm(total=len(dataloader)) as pbar:
                    for batch in dataloader:
                        loss, pairwise_loss, correct, pairs = run_iter(batch, num_modalities, model, device, mode, temperature, batch_size, ij)
                        total_loss += loss.item()
                        total_pairwise_loss += pairwise_loss
                        total_correct += correct
                        total_n += batch[0][0].size(0)
                        total_pairs += pairs

                        loss /= batch[0][0].size(0)
                        if split == "pretrain":
                            optim.zero_grad()
                            loss.backward()
                            optim.step()

                        if temperature < 0:
                            with torch.no_grad():
                                temperature.fill_(0)

                        if mode == "pairwise":
                            pbar.set_postfix_str(
                                f"Loss: {total_loss / total_n:.5f} ({loss:.5f}); " +
                                "Acc: {}; ".format(" ".join(map("{:.1f}".format, [100 * (total_correct[i, j] + total_correct[j, i]) / 2 / total_pairs[i, j] for i in range(len(modality_types)) for j in range(i + 1, len(modality_types))]))) +
                                f"Temperature: {temperature.item():.3f}"
                            )
                            if config["use_wandb"] and count_iter % log_interval == 0:
                                wandb.log({
                                    "Pairwise_train_loss": loss.item(),
                                    "Pairwise_total_loss": total_loss / total_n,
                                    "Pairwise_temperature": temperature.item(),
                                    **{f"Pairwise_acc_{i}_{j}": 100 * (total_correct[i, j] + total_correct[j, i]) / 2 / total_pairs[i, j] for i in range(len(modality_types)) for j in range(i + 1, len(modality_types))}
                                }, step=count_iter)
                        elif mode == "leave_one_out":
                            pbar.set_postfix_str(
                                f"Loss: {total_loss / total_n:.5f} ({loss:.5f}); " +
                                "Acc: {}; ".format(" ".join(map("{:.1f}".format, [100 * (total_correct[i, 0] + total_correct[i, 1]) / (total_pairs[i, 0] + total_pairs[i, 1]) for i in range(len(modality_types))]))) +
                                f"Temperature: {temperature.item():.3f}"
                            )
                            if config["use_wandb"] and count_iter % log_interval == 0:
                                wandb.log({
                                    "LeaveOneOut_train_loss": loss.item(),
                                    "LeaveOneOut_total_loss": total_loss / total_n,
                                    "LeaveOneOut_temperature": temperature.item(),
                                    **{f"LeaveOneOut_acc_{i}_0": 100 * total_correct[i, 0] / total_pairs[i, 0] for i in range(len(modality_types))},
                                    **{f"LeaveOneOut_acc_{i}_1": 100 * total_correct[i, 1] / total_pairs[i, 1] for i in range(len(modality_types))}
                                }, step=count_iter)

                        if (count_iter % config["save_iter"]) == 0:
                            logger.info(f"Iteration {count_iter} reached. Saving a checkpoint...")
                            save = {
                                "epoch": epoch,
                                "temperature": temperature.item(),
                                "optim_dict": optim.state_dict(),
                                "scheduler_dict": scheduler.state_dict(),
                                "best_loss": best_loss,
                                "loss": loss, 
                                "state_dict": model.state_dict()
                            }
                            torch.save(save, os.path.join(output, "checkpoint.pt"))
                            save_data(config, os.path.join(output, "config.json"))

                        if (count_iter % config["eval_iter"]) == 0:
                            logger.info(f"Iteration {count_iter} reached. Running validation now...")
                            if mode == "pairwise":
                                total_loss_val = 0.
                                total_pairwise_loss_val = np.zeros((num_modalities, num_modalities), dtype=float)
                                total_correct_val = np.zeros((num_modalities, num_modalities), dtype=int)
                                total_n_val = 0
                                total_pairs_val = np.zeros((num_modalities, num_modalities), dtype=int)
                            elif mode == "leave_one_out":
                                total_loss_val = 0.
                                total_pairwise_loss_val = np.zeros((num_modalities, 2), dtype=float)
                                total_correct_val = np.zeros((num_modalities, 2), dtype=int)
                                total_n_val = 0
                                total_pairs_val = np.zeros((num_modalities, 2), dtype=int)

                            dataloader_val = torch.utils.data.DataLoader(dataset["validation"], batch_size=batch_size, num_workers=4, shuffle=False, collate_fn=collate_fn)
                            model.eval()  
                            with torch.no_grad():
                                with tqdm.tqdm(total=len(dataloader_val)) as pbar_val:
                                    for batch_val in dataloader_val:
                                        loss, pairwise_loss, correct, pairs = run_iter(batch_val, num_modalities, model, device, mode, temperature, batch_size, ij)
                                        total_loss_val += loss.item()
                                        total_pairwise_loss_val += pairwise_loss
                                        total_correct_val += correct
                                        total_n_val += batch_val[0][0].size(0)
                                        total_pairs_val += pairs

                                        pbar_val.update()

                                if mode == "pairwise":
                                    pbar.set_postfix_str(
                                        f"Validation Loss: {total_loss_val / total_n_val:.5f}; " +
                                        "Validation Acc: {}; ".format(" ".join(map("{:.1f}".format, [100 * (total_correct_val[i, j] + total_correct_val[j, i]) / 2 / total_pairs_val[i, j] for i in range(len(modality_types)) for j in range(i + 1, len(modality_types))])))
                                    )
                                    if config["use_wandb"] and count_iter % log_interval == 0:
                                        wandb.log({
                                            "Pairwise_val_loss": total_loss_val / total_n_val,
                                            **{f"Pairwise_val_acc_{i}_{j}": 100 * (total_correct_val[i, j] + total_correct_val[j, i]) / 2 / total_pairs_val[i, j] for i in range(len(modality_types)) for j in range(i + 1, len(modality_types))}
                                        }, step=count_iter)
                                elif mode == "leave_one_out":
                                    pbar.set_postfix_str(
                                        f"Validation Loss: {total_loss_val / total_n_val:.5f}; " +
                                        "Validation Acc: {}; ".format(" ".join(map("{:.1f}".format, [100 * (total_correct_val[i, 0] + total_correct_val[i, 1]) / (total_pairs_val[i, 0] + total_pairs_val[i, 1]) for i in range(len(modality_types))])))
                                    )
                                    if config["use_wandb"] and count_iter % log_interval == 0:
                                        wandb.log({
                                            "LeaveOneOut_val_loss": total_loss / total_n,
                                            **{f"LeaveOneOut_val_acc_{i}_0": 100 * total_correct_val[i, 0] / total_pairs_val[i, 0] for i in range(len(modality_types))},
                                            **{f"LeaveOneOut_val_acc_{i}_1": 100 * total_correct_val[i, 1] / total_pairs_val[i, 1] for i in range(len(modality_types))}
                                        }, step=count_iter)
                            model.train()
                            batch_val = None
                        count_iter += 1
                        pbar.update()
            if mode == "pairwise":
                f.write("{}\t{}\t".format(epoch, split))
                f.write(((len(ij) + 1) * "{:.5f}\t").format(total_loss / total_n, *[total_pairwise_loss[i, j] / total_pairs[i, j] for (i, j) in ij]))
                f.write((len(ij) * "{:.3f}\t").format(*[100 * total_correct[i, j] / total_pairs[i, j] for (i, j) in ij]))
                f.write("{:.5f}\n".format(temperature.item()))
            elif mode == "leave_one_out":
                f.write("{}\t{}\t".format(epoch, split))
                f.write(((num_modalities  * 2 + 1) * "{:.5f}\t").format(total_loss / total_n, *[total_pairwise_loss[i, j] / total_pairs[i, j] for i in range(num_modalities ) for j in [0, 1]]))
                f.write(((num_modalities  * 2) * "{:.3f}\t").format(*[100 * total_correct[i, j] / total_pairs[i, j] for i in range(num_modalities ) for j in [0, 1]]))
                f.write("{:.5f}\n".format(temperature.item()))
            f.flush()

            scheduler.step()

            loss = total_loss / total_n 
            is_best = (loss < best_loss)
            if is_best:
                best_loss = loss

            save = {
                "epoch": epoch,
                "temperature": temperature.item(),
                "optim_dict": optim.state_dict(),
                "scheduler_dict": scheduler.state_dict(),
                "best_loss": best_loss,
                "loss": loss, 
                "state_dict": model.state_dict()
            }

            if is_best:
                torch.save(save, os.path.join(output, "best.pt"))
            torch.save(save, os.path.join(output, "checkpoint.pt"))
            save_data(config, os.path.join(output, "config.json"))

    if config["use_wandb"]:
        wandb.finish()
    logger.info("Finished Training!!")


class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


if __name__ == '__main__':
    pretrain()