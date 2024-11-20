# code_MEGAPOLI_Foundation_Model

- [code\_MEGAPOLI\_Foundation\_Model](#code_megapoli_foundation_model)
  - [Introduction](#introduction)
  - [Scripts and Data](#scripts-and-data)
    - [Prerequisite](#prerequisite)
    - [Scripts](#scripts)

## Introduction
This repository is a supplementary to the manuscript **"Integrating Simulations and Observations: A Foundation Model for Estimating Aerosol Mixing State Index"**

The objective of this project are:
- Pre-train a foundation model for aerosol mixing state prediction using PartMC-MOSAIC simulation data.
- Fine-tune the pre-trained foundation model with MEGAPOLI observational data.
- Analyze the impact of data scarcity on the performance of the fine-tuned model and input feature importance.

## Scripts and Data

### Prerequisite

- If you do not have the **"[conda](https://docs.conda.io/en/latest/)"** system
  
```bash
# Download and install conda
$ wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
$ chmod +x Miniconda3-latest-Linux-x86_64.sh
$ ./Miniconda3-latest-Linux-x86_64.sh
# Edit .bash_profile or .bashrc
PATH=$PATH:$HOME/.local/bin:$HOME/bin:$HOME/miniconda3/bin
# Activate the conda system
$source .bash_profile
# OR source .bashrc
```

- Create and activate your own conda environment

```bash
# Create an environment "partmc" and install the necessary packages
conda env create -f environment.yml
# Activate the "partmc" environment
conda activate partmc
```


### Scripts

| Tasks             | Folders            | Fig or Tab in paper                                          | Fig or Tab in preprint |
| ------------------------------ | ------------------ | -------------------------------------------- | ---------------------- |
| Pre-trained foundation model hyperparameter | 1_Pre_trained_foundation_model_hyperparameter hyperparameter      |                                                              |                        |
| Pre-trained foundation model development  | 2_Pre_train_Foundation_model      |                                         | Fig S3                  |
| Fine-tuned foundation model development (various fine-tuning training data size) | 3_Fine_tune_different_data_size        |                                                              |        Fig 2 &3(a)(b)                |
| Fine-tuned foundation model development (various input feature size)     | 4_Fine_tune_different_input_feature_size |  | Fig 3(c)          |    
| Data analysis| 5_Figure_plot| |  Fig 2,3,S1,S2,S3 and S4 |
