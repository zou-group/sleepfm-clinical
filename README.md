# SleepFM-Clinical

## 🔥 News & Updates

### 📰 Publication
- **SleepFM** has been published in **Nature Medicine**  
  👉 https://www.medrxiv.org/content/10.1101/2025.02.04.25321675v1

### 🤖 Model Releases
- **Pretrained Base Model**  
  `sleepfm/checkpoints/model_base`
- **Finetuned Disease Prediction Model**  
  `sleepfm/checkpoints/model_diagnosis`
- **Finetuned Sleep Staging Model**  
  `sleepfm/checkpoints/model_sleep_staging`

### 📊 Dataset Release
- **Stanford Sleep Dataset**  
  Public release of the Stanford of data used to pretrain **SleepFM**  
  👉 https://bdsp.io/content/08vg8vqv2wdtwonc1ddy/1.0

### 🚀 Demos
- 📓 **End-to-End Inference Demo**  
  `notebooks/demo.ipynb`

## 📖 Introduction

Sleep is a fundamental biological process with broad implications for physical and mental health, yet its complex relationship with disease remains poorly understood. Polysomnography (PSG), the gold standard for sleep analysis, captures rich physiological signals but remains underutilized due to challenges in standardization, generalizability, and multimodal integration. To address these limitations, we developed SleepFM, a multimodal sleep foundation model trained with a novel contrastive learning approach that accommodates multiple PSG montages—the specific arrangements of electrodes and sensors used to record physiological signals during sleep. Trained on a curated dataset of over 585,000 hours of PSG recordings from approximately 65,000 participants across multiple cohorts, SleepFM produces latent sleep representations that capture the physiological and temporal structure of sleep and enable accurate prediction of future disease risk. SleepFM achieved a C-Index of at least 0.75 (Bonferroni-corrected p < 0.01) for 130 conditions, including all-cause mortality (C-Index: 0.84), dementia (0.85), myocardial infarction (0.81), heart failure (0.80), chronic kidney disease (0.79), stroke (0.78), and atrial fibrillation (0.78). Moreover, the model demonstrates strong transfer learning performance on a dataset from the Sleep Heart Health Study (SHHS), a dataset that was excluded from pretraining, and performs competitively with specialized sleep-staging models such as U-Sleep and YASA on common sleep analysis tasks, achieving mean F1 scores of 0.70–0.78 for sleep staging and accuracies of 0.69 and 0.87 for classifying sleep apnea severity and presence. This work shows that foundation models can extract clinically meaningful features from multi-modal sleep recordings, enabling scalable, label-efficient analysis and disease prediction.

# 📖 Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Licence](#license)

<a name="installation"/>

# 💿 Installation

Follow the steps below to set up an environment for running **SleepFM**.

We recommend using **Python 3.10**. All required packages and their specific versions are listed in the [requirements.txt](https://github.com/zou-group/sleepfm-clinical/blob/main/requirements.txt) file. Installing all dependencies should take only **2–3 minutes**.

### 🖥️ System Requirements

SleepFM was developed and tested on Linux systems with the following configuration:

- **GPU**: NVIDIA A40, A100, and RTX 2080 Ti
- **CUDA**: 12.4
- **CPU**: 8 cores recommended
- **RAM**: At least 32 GB
- **OS**: CentOS Linux 7.9.2009 (Core)

Although optimized for **A40/A100** GPUs, the model can be run on smaller GPUs (e.g., **RTX 2080 Ti**) by reducing the batch size. For smooth performance during preprocessing and training, we recommend using at least **8 CPU cores**.

### 🚀 Demo Run

This codebase includes a demo using the **MESA** dataset. On an **NVIDIA A40**, pretraining on MESA for one epoch takes approximately **1 hour**, and fine-tuning for one epoch takes about **1 minute**. Note that **MESA is significantly smaller** than the full dataset used in our main experiments.




```bash
git clone https://github.com/zou-group/sleepfm-clinical.git
cd sleepfm-clinical
conda env create -f env.yml
conda activate sleepfm_env
```

<a name="usage"/>

# 👩‍💻 Usage

This codebase will serve as a framework that you can adapt to your dataset for pretraining and testing. Below, we outline the steps to pretrain and adapt the model on a publicly available dataset called [MESA](https://sleepdata.org/datasets/mesa). Please keep in mind that this dataset is small and will most likely not yield optimal results.

**Note**: Please make sure to download the dataset with in your local path, with dataset name, `mesa`. Later on, we will need this path. 

## Preprocessing Dataset

PSG files may be stored in different formats. Here, we specifically provide scripts to process .EDF file format.

- **Step 0:** `preprocessing/preprocessing.py`
  - This script converts .EDF file into .hdf5 files with is the format that the model will expect below. 


## Pretraining

Note that we provide with dataset split as json file here: `configs/dataset_split.json`. We also provide with different channel groups within a modality: `configs/channel_groups.json`.

- **Step 1:** `sleepfm/pipeline/pretrain.py`
  - This script has our main pretraining config. Its corresponding config file is inside `configs/config_set_transformer_contrastive.yaml`, where you will set all the parameters and data path. This step will roughly take about an hour for an epoch on `MESA`. 
- **Step 2:** `sleepfm/pipeline/generate_embeddings.py`
  - After pretraining our model, we want to generate the embeddings for train/valid/test so that we can train a model for downstream classification. We do sleep stage classification here. This step will roughly take few minutes on `MESA`. 

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

- **Step 3:** `sleepfm/pipelinefinetune_sleep_staging.py`
  - This will finetune the pretrained model on sleep stage classification task. Please make sure to check config `configs/config_finetune_sleep_events.yaml`. This step roughly takes less than a minute on `MESA`. 

- **Step 4:** `sleepfm/pipeline/evaluate_sleep_staging.py`
  - This will evaluate the model on test set. This step only takes few seconds on `MESA`.
 

For disease prediction task:

- **Step 3:** `sleepfm/pipeline/finetune_diagnosis_coxph.py`
  - This will finetune the pretrained model on disease prediction task, using CoxPH loss function. Note that you will need to provide your own data, and set up dataloaders. Please see corresponding config `sleepfm/configs/config_finetune_diagnosis_coxph.yaml`


## BibTeX

```bibtex
@article{thapa2025multimodal,
  title={A Multimodal Sleep Foundation Model Developed with 500K Hours of Sleep Recordings for Disease Predictions},
  author={Thapa, Rahul and Kj{\ae}r, Magnus Ruud and He, Bryan and Covert, Ian and Moore, Hyatt and Hanif, Umaer and Ganjoo, Gauri and Westover, Brandon M and Jennum, Poul and Brink-Kj{\ae}r, Andreas and others},
  journal={medRxiv},
  pages={2025--02},
  year={2025},
  publisher={Cold Spring Harbor Laboratory Press}
}
```

## License

[MIT License](LICENSE)
