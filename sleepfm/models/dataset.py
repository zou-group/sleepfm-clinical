from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import h5py
import random
import torch
from tqdm import tqdm
import multiprocessing
import time
import sys
sys.path.append("../")
from utils import load_data, save_data
import pandas as pd
import json
from loguru import logger
from einops import rearrange


def index_file_helper(args):
    file_path, channel_like, chunk_size, channel_groups, modality_types = args
    file_index_map = []
    modality_to_channels = {modality_type: [] for modality_type in modality_types}
    try:
        with h5py.File(file_path, 'r', rdcc_nbytes = 300 * 512 * 8 * 2) as hf:
            dset_names = []
            for dset_name in hf.keys():
                if not channel_like or dset_name in channel_like:
                    if isinstance(hf[dset_name], h5py.Dataset):
                        dset_names.append(dset_name)
                        if dset_name in channel_groups["BAS"]:
                            modality_to_channels["BAS"].append(dset_name)
                        if dset_name in channel_groups["RESP"]:
                            modality_to_channels["RESP"].append(dset_name)
                        if dset_name in channel_groups["EKG"]:
                            modality_to_channels["EKG"].append(dset_name)
                        if dset_name in channel_groups["EMG"]:
                            modality_to_channels["EMG"].append(dset_name)
            flag = True
            for modality, channels in modality_to_channels.items():
                if len(channels) == 0:
                    flag = False
                    break
            if flag:
                num_samples = hf[dset_name].shape[0]
                num_chunks = num_samples // chunk_size
                for chunk_start in range(0, num_chunks * chunk_size, chunk_size):
                    file_index_map.append((file_path, dset_names, chunk_start))
    except (OSError, AttributeError) as e:
        with open("problem_hdf5.txt", "a") as f:
            f.write(f"Error processing file {file_path}: {str(e)}\n")
    return file_index_map

def index_files(hdf5_paths, channel_like, samples_per_chunk, num_workers, channel_groups=None, modality_types=None):
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(index_file_helper, [(path, channel_like, samples_per_chunk, channel_groups, modality_types) for path in hdf5_paths]), total=len(hdf5_paths), desc="Indexing files", position=0, leave=True))
    return [item for sublist in results for item in sublist]


class SetTransformerDataset(Dataset):
    def __init__(self, 
                 config,
                 channel_groups,
                 hdf5_paths=[],
                 split="pretrain"):

        self.config = config
        self.channel_groups = channel_groups
        channel_like = []
        for modality_type in config["modality_types"]:
            channel_like += channel_groups[modality_type]
        channel_like = set(channel_like)

        data_path = config["data_path"]

        if len(hdf5_paths) == 0:
            hdf5_paths = load_data(config["split_path"])[split]
        
        hdf5_paths = [os.path.join(data_path, path) for path in hdf5_paths]

        if split in ["pretrain"]:
            random.shuffle(hdf5_paths)

        if config["max_files"]:
            self.hdf5_paths = hdf5_paths[:config["max_files"]]
        else:
            self.hdf5_paths = hdf5_paths

        if split == "validation":
            self.hdf5_paths = self.hdf5_paths[:config["val_size"]]
        
        self.samples_per_chunk = config["sampling_duration"] * 60 * config["sampling_freq"]  
        
        # Use multiprocessing to index files in parallel
        self.index_map = index_files(self.hdf5_paths, channel_like, self.samples_per_chunk, config["num_workers"], channel_groups=self.channel_groups, modality_types=config["modality_types"])

        # random.shuffle(self.hdf5_paths)
        # self.index_map = sorted(self.index_map, key=lambda x: (x[1], x[2]))
        self.total_len = len(self.index_map)
        self.modalities_length = []
        for modality_type in self.config["modality_types"]:
            self.modalities_length.append(self.config[f'{modality_type}_CHANNELS'])

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        file_path, dset_names, chunk_start = self.index_map[idx]

        modality_to_channels = {modality_type: [] for modality_type in self.config["modality_types"]}
        for dset_name in dset_names:
            if dset_name in self.channel_groups["BAS"]:
                modality_to_channels["BAS"].append(dset_name)
            if dset_name in self.channel_groups["RESP"]:
                modality_to_channels["RESP"].append(dset_name)
            if dset_name in self.channel_groups["EKG"]:
                modality_to_channels["EKG"].append(dset_name)
            if dset_name in self.channel_groups["EMG"]:
                modality_to_channels["EMG"].append(dset_name)

        target = []
        with h5py.File(file_path, 'r', rdcc_nbytes=300 * 512 * 8 * 2) as hf:
            for modality_type in self.config["modality_types"]:
                num_channels = self.config[f"{modality_type}_CHANNELS"]
                data = np.zeros((len(modality_to_channels[modality_type]), self.samples_per_chunk))
                ds_names = modality_to_channels[modality_type]
                for idx, ds_name in enumerate(ds_names):
                    signal = hf[ds_name][chunk_start:chunk_start+self.samples_per_chunk]
                    data[idx] = signal
                target.append(torch.from_numpy(data).float())
        return target, file_path, dset_names, chunk_start, self.modalities_length


def collate_fn(batch):
    # Determine the number of modalities

    file_paths = [batch[i][1] for i in range(len(batch))]
    dset_names_list = [batch[i][2] for i in range(len(batch))]
    chunk_starts = [batch[i][3] for i in range(len(batch))]
    batch = [batch[i][0] for i in range(len(batch))]

    num_modalities = len(batch[0])
    
    # Initialize lists to hold padded data and masks for each modality
    padded_batch_list = [[] for _ in range(num_modalities)]
    mask_list = [[] for _ in range(num_modalities)]
    
    # Iterate over each modality
    for modality_index in range(num_modalities):
        max_channels = max(data[modality_index].shape[0] for data in batch)
        
        for data in batch:
            modality_data = data[modality_index]
            channels, length = modality_data.shape
            pad_channels = max_channels - channels
            
            # Create mask: 0 for real values, 1 for padded values
            mask = torch.cat((torch.zeros(channels), torch.ones(pad_channels)), dim=0)
            mask_list[modality_index].append(mask)
            
            # Pad the channel dimension
            pad_channel_tensor = torch.zeros((pad_channels, length))
            modality_data = torch.cat((modality_data, pad_channel_tensor), dim=0)
            
            padded_batch_list[modality_index].append(modality_data)
        
        # Stack the padded data and masks for the current modality
        padded_batch_list[modality_index] = torch.stack(padded_batch_list[modality_index])
        mask_list[modality_index] = torch.stack(mask_list[modality_index])
    
    return padded_batch_list, mask_list, file_paths, dset_names_list, chunk_starts


class SleepEventClassificationDataset(Dataset):
    def __init__(self, 
                 config,
                 channel_groups,
                 hdf5_paths=[],
                 split="train"):

        self.config = config
        self.max_channels = self.config["max_channels"]
        self.context = int(self.config["context"])
        self.channel_like = self.config["channel_like"]

        data_path = config["data_path"]

        labels_path = self.config["labels_path"]
        dataset = self.config["dataset"]
        dataset = dataset.split(",")

        label_files = []

        for dataset_name in dataset:
            label_files += glob.glob(os.path.join(labels_path, dataset_name, "**", "*.csv"), recursive=True)

        # label_files = [label_file for label_file in os.listdir(labels_path) if label_file.endswith(".csv")]

        hdf5_paths = load_data(config["split_path"])[split]
        hdf5_paths = [os.path.join(data_path, path) for path in hdf5_paths]
        study_ids = set([os.path.basename(label_file).split(".")[0] for label_file in label_files])

        hdf5_paths = [f for f in hdf5_paths if os.path.exists(f)]
        hdf5_paths = [f for f in hdf5_paths if f.split("/")[-1].split(".")[0] in study_ids]

        hdf5_paths_ids = set([os.path.basename(hdf5_path).split(".")[0] for hdf5_path in hdf5_paths])

        hdf5_paths_new = []
        for dataset_name in dataset:
            hdf5_paths_new += glob.glob(os.path.join(config["model_path"], dataset_name, "**", "*.hdf5"), recursive=True)
        
        hdf5_paths_new = [item for item in hdf5_paths_new if os.path.basename(item).split(".")[0] in hdf5_paths_ids]
        hdf5_paths = hdf5_paths_new
        hdf5_paths = [f for f in hdf5_paths if os.path.exists(f)]

        if config["max_files"]:
            hdf5_paths = hdf5_paths[:config["max_files"]]
        else:
            hdf5_paths = hdf5_paths

        labels_dict = {
            os.path.basename(item).split(".")[0]: item for item in label_files
        }

        if self.context == -1:
            self.index_map = [(path, labels_dict[path.split("/")[-1].split(".")[0]], -1) for path in hdf5_paths]
        else:
            self.index_map = []
            loop = tqdm(hdf5_paths[:], total=len(hdf5_paths), desc=f"Indexing {split} data")
            for hdf5_file_path in loop:
                file_prefix = os.path.basename(hdf5_file_path).split(".")[0]
                with h5py.File(hdf5_file_path, "r") as file:
                    dataset_names = list(file.keys())[:]
                    dataset_name = dataset_names[0]
                    dataset_length = file[dataset_name].shape[0]
                    for i in range(0, dataset_length, self.context):
                        self.index_map.append((hdf5_file_path, labels_dict[file_prefix], i))           
            
        logger.info(f"Number of files in {split} set: {len(hdf5_paths)}")
        logger.info(f"Number of files to be processed in {split} set: {len(self.index_map)}")
        self.total_len = len(self.index_map)
        self.max_seq_len = config["model_params"]["max_seq_length"]

    def __len__(self):
        return self.total_len

    def get_index_map(self):
        return self.index_map

    def __getitem__(self, idx):
        hdf5_path, label_path, start_index = self.index_map[idx]
        labels_df = pd.read_csv(label_path)
        labels_df["StageNumber"] = labels_df["StageNumber"].replace(-1, 0)

        #### Temp
        # labels_df["StageNumber"] = labels_df["StageNumber"].apply(lambda x: 1 if x > 0 else 0)
        
        y_data = labels_df["StageNumber"].to_numpy()

        x_data = []
        with h5py.File(hdf5_path, 'r') as hf:
            dset_names = list(hf.keys())[:]
            if self.context != -1:
                y_data = y_data[start_index:start_index+self.context]
            for dataset_name in dset_names:
                if dataset_name in self.channel_like:
                    if self.context == -1:
                        x_data.append(hf[dataset_name][:])
                    else:
                        x_data_in = hf[dataset_name][start_index:start_index+self.context]
                        x_data.append(x_data_in)

        if not x_data:
            # Skip this data point if x_data is empty
            return self.__getitem__((idx + 1) % self.total_len)

        # Convert x_data list to a single numpy array
        x_data = np.array(x_data)

        # Convert x_data to tensor
        x_data = torch.tensor(x_data, dtype=torch.float32)
        y_data = torch.tensor(y_data, dtype=torch.float32)
        min_length = min(x_data.shape[1], len(y_data))
        x_data = x_data[:, :min_length, :]
        y_data = y_data[:min_length]

        return x_data, y_data, self.max_channels, self.max_seq_len, hdf5_path


def sleep_event_finetune_full_collate_fn(batch):
    x_data, y_data, max_channels_list, max_seq_len_list, hdf5_path_list = zip(*batch)

    num_channels = max(max_channels_list)

    max_seq_len_temp = max([item.size(1) for item in x_data])
    # Determine the max sequence length for padding
    if max_seq_len_list[0] is None:
        max_seq_len = max_seq_len_temp
    else:
        max_seq_len = min(max_seq_len_temp, max_seq_len_list[0])

    padded_x_data = []
    padded_y_data = []
    padded_mask = []

    for x_item, y_item in zip(x_data, y_data):
        # Get the shape of x_item
        c, s, e = x_item.size()
        c = min(c, num_channels)
        s = min(s, max_seq_len)  # Ensure the sequence length doesn't exceed max_seq_len

        # Create a padded tensor and a mask tensor for x_data
        padded_x_item = torch.zeros((num_channels, max_seq_len, e))
        mask = torch.ones((num_channels, max_seq_len))

        # Copy the actual data to the padded tensor and set the mask for real data
        padded_x_item[:c, :s, :e] = x_item[:c, :s, :e]
        mask[:c, :s] = 0  # 0 for real data, 1 for padding

        # Pad y_data with zeros to match max_seq_len
        padded_y_item = torch.zeros(max_seq_len)
        padded_y_item[:s] = y_item[:s]

        # Append padded items to lists
        padded_x_data.append(padded_x_item)
        padded_y_data.append(padded_y_item)
        padded_mask.append(mask)

    # Stack all tensors into a batch
    x_data = torch.stack(padded_x_data)
    y_data = torch.stack(padded_y_data)
    padded_mask = torch.stack(padded_mask)
    
    return x_data, y_data, padded_mask, hdf5_path_list


def sleep_event_finetune_full_collate_fn(batch):
    x_data, y_data, max_channels_list, max_seq_len_list, hdf5_path_list = zip(*batch)

    num_channels = max(max_channels_list)

    max_seq_len_temp = max([item.size(1) for item in x_data])
    # Determine the max sequence length for padding
    if max_seq_len_list[0] is None:
        max_seq_len = max_seq_len_temp
    else:
        max_seq_len = min(max_seq_len_temp, max_seq_len_list[0])

    padded_x_data = []
    padded_y_data = []
    padded_mask = []

    for x_item, y_item in zip(x_data, y_data):
        # Get the shape of x_item
        c, s, e = x_item.size()
        c = min(c, num_channels)
        s = min(s, max_seq_len)  # Ensure the sequence length doesn't exceed max_seq_len

        # Create a padded tensor and a mask tensor for x_data
        padded_x_item = torch.zeros((num_channels, max_seq_len, e))
        mask = torch.ones((num_channels, max_seq_len))

        # Copy the actual data to the padded tensor and set the mask for real data
        padded_x_item[:c, :s, :e] = x_item[:c, :s, :e]
        mask[:c, :s] = 0  # 0 for real data, 1 for padding

        # Pad y_data with zeros to match max_seq_len
        padded_y_item = torch.zeros(max_seq_len)
        padded_y_item[:s] = y_item[:s]

        # Append padded items to lists
        padded_x_data.append(padded_x_item)
        padded_y_data.append(padded_y_item)
        padded_mask.append(mask)

    # Stack all tensors into a batch
    x_data = torch.stack(padded_x_data)
    y_data = torch.stack(padded_y_data)
    padded_mask = torch.stack(padded_mask)
    
    return x_data, y_data, padded_mask, hdf5_path_list