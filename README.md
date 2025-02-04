# SleepFM-Clinical

## 🔥 News
- [Our paper](todo) will be out on bioarxiv soon.

## 📖 Introduction

Coming soon!


# 📖 Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Licence](#license)

<a name="installation"/>

# 💿 Installation

Please use the following steps to create an environment for running SleepFM

```bash
git clone https://github.com/zou-group/sleepfm-clinical.git
cd sleepfm-clinical
conda env create -f env.yml
conda activate sleepfm_env
```

<a name="usage"/>

# 👩‍💻 Usage

*This is a research code. Here, we provide our pretraining pipeline with a publicly available dataset, as we cannot release our internal pretraining dataset at the moment.*

This codebase will serve as a framework that you can adapt to your dataset for pretraining and testing. Below, we outline the steps to pretrain and adapt the model on a publicly available dataset called [MESA](https://sleepdata.org/datasets/mesa). Please keep in mind that this dataset is small and will most likely not yield optimal results.

## Downloading Dataset

We are working with MESA dataset as a usecase here. 

- Follow the [link](https://sleepdata.org/datasets/mesa). You may need to create a physionet account. 

**Note**: Please make sure to download the dataset with in your local path, with dataset name, `mesa`. Later on, we will need this path. 


## Preprocessing Dataset

PSG files may be stored in different formats. Here, we specifically provide scripts to process .EDF file format.

- **Step 0:** `preprocessing/preprocessing.py`
  - This script converts .EDF file into .hdf5 files with is the format that the model will expect below. 


## Pretraining

Note that we provide with dataset split as json file here: `configs/dataset_split.json`. We also provide with different channel groups within a modality: `configs/channel_groups.json`.

- **Step 1:** `pipeline/pretrain.py`
  - This script has our main pretraining config. Its corresponding config file is inside `configs/config_set_transformer_contrastive.yaml`, where you will set all the parameters and data path. 
- **Step 2:** `pipeline/generate_embeddings.py`
  - After pretraining our model, we want to generate the embeddings for train/valid/test so that we can train a model for downstream classification. We do sleep stage classification here. 

## Evaluation

Note: These evaluation results will not match the ones that we have in our paper as this is a small dataset. This step does not require GPU support. 

You should also have extracted the sleep stage labels, which should look like this:

```csv
Start,Stop,StageName,StageNumber
0.0,5190.0,Wake,0
0.0,5190.0,Wake,0,
0.0,5190.0,Wake,0
```

These labels files are stored inside a folder as such `<path>/mesa/mesa-sleep-0001.csv`. Note that `mesa-sleep-0001` is the filename that should correspond with the original `.EDF` file and `.hdf5` files. 

- **Step 3:** `finetune_sleep_staging.py`
  - This will finetune the pretrained model on sleep stage classification task. Please make sure to check config `configs/config_finetune_sleep_events.yaml`. 

- **Step 4:** `evaluate_sleep_staging.py`
  - This will evaluate the model on test set. 


## Model Checkpoint

Coming Soon!


## BibTeX

```bibtex
<todo>
```

## 🪪 License

[MIT License](LICENSE)
