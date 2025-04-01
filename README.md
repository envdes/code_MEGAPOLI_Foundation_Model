# code_MEGAPOLI_Foundation_Model

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14223751.svg)](https://doi.org/10.5281/zenodo.14223751)

- [code\_MEGAPOLI\_Foundation\_Model](#code_megapoli_foundation_model)
  - [Introduction](#introduction)
  - [Scripts and Data](#scripts-and-data)
    - [Prerequisite](#prerequisite)
    - [Scripts](#scripts)
    - [Data](#data)
    - [Model](#model)
  - [Acknowledgments](#acknowledgments)

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

| Tasks             | Folders            | Fig or Tab in paper                                          | 
| ------------------------------ | ------------------ | -------------------------------------------- |
| PartMC data extract |0_PartMC_data_extract   |   |
| Pre-trained foundation model hyperparameter | 1_Pre_trained_foundation_model_hyperparameter hyperparameter      |                                                              |                        |
| Pre-trained foundation model development  | 2_Pre_train_Foundation_model      |                                         |
| Fine-tuned foundation model development (various fine-tuning training data size) | 3_Fine_tune_different_data_size        |                                                              |        Fig 2 &3(a)(b)                |
| Fine-tuned foundation model development (various input feature size)     | 4_Fine_tune_different_input_feature_size |       
| Data analysis| 5_Figure_plot| |  


### Data
- PartMC data
  
| Folder                                                       | Comments                                                     | How to get it?                                               |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
|PartMC_train.csv  | PartMC simulation training data for pre-trained foundation model | Raw Partmc Data and [Pre_train_Foundation_model.ipynb](https://github.com/envdes/code_MEGAPOLI_Foundation_Model/blob/main/2_Pre_train_Foundation_model/Pre_train_Foundation_model.ipynb)  | 
|PartMC_valid.csv  | PartMC simulation validation data for pre-trained foundation model | Raw Partmc Data and [Pre_train_Foundation_model.ipynb](https://github.com/envdes/code_MEGAPOLI_Foundation_Model/blob/main/2_Pre_train_Foundation_model/Pre_train_Foundation_model.ipynb)  | 
|PartMC_test.csv  | PartMC simulation testing data for pre-trained foundation model | Raw Partmc Data and [Pre_train_Foundation_model.ipynb](https://github.com/envdes/code_MEGAPOLI_Foundation_Model/blob/main/2_Pre_train_Foundation_model/Pre_train_Foundation_model.ipynb)  | 


- MEGAPOLI data: MEGAPOLI observational data will be made available on request.

- Fine_tuned_Results_different_data_szie

| Folder                                                       | Comments                                                     | How to get it?                                               |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
|Fine_tuning_XX%Data.csv  | Chi estimation results from fine-tuned foundation model, here XX means training dataset is XX fraction of total MEGAPOLI data (XX *2 fraction of fine-tuning training dataset) | [Fine-tune_different_data_size.ipynb](https://github.com/envdes/code_MEGAPOLI_Foundation_Model/blob/main/3_Fine_tune_different%20_data_size/Fine-tune_different_data_size.ipynb)  | 
|Fine_tuning_XX%Data_test.csv |Chi estimation results from fine-tuned foundation model ('Swapped Temporal Order' experiment), here XX means training dataset is XX fraction of total MEGAPOLI data (XX *2 fraction of fine-tuning testing dataset), model performance evaluated on fine-tuning training dataset |[Fine-tune_different_data_size.ipynb](https://github.com/envdes/code_MEGAPOLI_Foundation_Model/blob/main/3_Fine_tune_different%20_data_size/Fine-tune_different_data_size.ipynb)
|AutoML_XX%Data.csv  | Chi estimation results from AutoML, here XX means training dataset is XX fraction of total MEGAPOLI data (XX *2 fraction of fine-tuning training dataset) | [AutoML.ipynb](https://github.com/envdes/code_MEGAPOLI_Foundation_Model/blob/main/3_Fine_tune_different%20_data_size/AutoML.ipynb)  | 
|AutoML_XX%Data_test.csv  | Chi estimation results from AutoML, here XX means training dataset is XX fraction of total MEGAPOLI data ('Swapped Temporal Order' experiment) (XX *2 fraction of fine-tuning testing dataset), model performance evaluated on fine-tuning training dataset | [AutoML.ipynb](https://github.com/envdes/code_MEGAPOLI_Foundation_Model/blob/main/3_Fine_tune_different%20_data_size/AutoML.ipynb)  |
|LR_XX%Data.csv  | Chi estimation results from Linear regression, here XX means training dataset is XX fraction of total MEGAPOLI data (XX *2 fraction of fine-tuning training dataset) | [LR.ipynb](https://github.com/envdes/code_MEGAPOLI_Foundation_Model/blob/main/3_Fine_tune_different%20_data_size/LR.ipynb)  | 
|LR_XX%Data_test.csv  | Chi estimation results from Linear regression ('Swapped Temporal Order' experiment), here XX means training dataset is XX fraction of total MEGAPOLI data (XX *2 fraction of fine-tuning testing dataset), model performance evaluated on fine-tuning training dataset  | [LR.ipynb](https://github.com/envdes/code_MEGAPOLI_Foundation_Model/blob/main/3_Fine_tune_different%20_data_size/LR.ipynb)  | 

- Fine_tuned_Results_different_input_feature

| Folder                                                       | Comments                                                     | How to get it?                                               |
| -------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
|Fine_tuning_DropAero_Data.csv  | Input feature drop aerosol subset | [Fine-tune_different_input_size.ipynb](https://github.com/envdes/code_MEGAPOLI_Foundation_Model/blob/main/4_Fine_tune_different_input_feature_size/Fine-tune_different_input_size.ipynb)  | 
|Fine_tuning_DropAllGas_Data.csv  | Input feature drop Non-VOC and VOC gases subset | [Fine-tune_different_input_size.ipynb](https://github.com/envdes/code_MEGAPOLI_Foundation_Model/blob/main/4_Fine_tune_different_input_feature_size/Fine-tune_different_input_size.ipynb)  | 
|Fine_tuning_DropEnv_Data.csv  | Input feature drop environment subset | [Fine-tune_different_input_size.ipynb](https://github.com/envdes/code_MEGAPOLI_Foundation_Model/blob/main/4_Fine_tune_different_input_feature_size/Fine-tune_different_input_size.ipynb)  | 
|Fine_tuning_DropNonVOC_Data.csv  | Input feature drop Non-VOC gases subset | [Fine-tune_different_input_size.ipynb](https://github.com/envdes/code_MEGAPOLI_Foundation_Model/blob/main/4_Fine_tune_different_input_feature_size/Fine-tune_different_input_size.ipynb)  | 
|Fine_tuning_DropVOC_Data.csv  | Input feature drop VOC gases subset | [Fine-tune_different_input_size.ipynb](https://github.com/envdes/code_MEGAPOLI_Foundation_Model/blob/main/4_Fine_tune_different_input_feature_size/Fine-tune_different_input_size.ipynb)  | 
|Fine_tuning_onlyAero_Data.csv  | Input feature only consider aerosol subset | [Fine-tune_different_input_size.ipynb](https://github.com/envdes/code_MEGAPOLI_Foundation_Model/blob/main/4_Fine_tune_different_input_feature_size/Fine-tune_different_input_size.ipynb)  | 


- Pre_training_data_volume

| Folder                                                       | Comments                                                     | How to get it?                                               |
| -------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
|Pre_trained_20%_after_fine_tune.csv  | Used 20% PartMC data to train the pre-trained foundation model, fine-tuned by fine-tuning training dataset and evaluated on fine-tuning testing dataset | [Pre_train_Foundation_model.ipynb](https://github.com/envdes/code_MEGAPOLI_Foundation_Model/blob/main/2_Pre_train_Foundation_model/Pre_train_Foundation_model.ipynb)  | 
|Pre_trained_50%_after_fine_tune.csv  | Used 50% PartMC data to train the pre-trained foundation model, fine-tuned by fine-tuning training dataset and evaluated on fine-tuning testing dataset | [Pre_train_Foundation_model.ipynb](https://github.com/envdes/code_MEGAPOLI_Foundation_Model/blob/main/2_Pre_train_Foundation_model/Pre_train_Foundation_model.ipynb)  | 
|Pre_trained_90%_after_fine_tune.csv  | Used 90% PartMC data to train the pre-trained foundation model, fine-tuned by fine-tuning training dataset and evaluated on fine-tuning testing dataset | [Pre_train_Foundation_model.ipynb](https://github.com/envdes/code_MEGAPOLI_Foundation_Model/blob/main/2_Pre_train_Foundation_model/Pre_train_Foundation_model.ipynb)  | 

### Model
| Folder                                                       | Comments                                                     | How to get it?                                               |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
|Foundation_Model_20%.pth |Foundation model pre-trained by 20% PartMC simulation data | [Pre_train_Foundation_model.ipynb](https://github.com/envdes/code_MEGAPOLI_Foundation_Model/blob/main/2_Pre_train_Foundation_model/Pre_train_Foundation_model.ipynb)|
|Foundation_Model_50%.pth |Foundation model pre-trained by 50% PartMC simulation data | [Pre_train_Foundation_model.ipynb](https://github.com/envdes/code_MEGAPOLI_Foundation_Model/blob/main/2_Pre_train_Foundation_model/Pre_train_Foundation_model.ipynb)|
|Foundation_Model.pth |Foundation model pre-trained by 90% PartMC simulation data | [Pre_train_Foundation_model.ipynb](https://github.com/envdes/code_MEGAPOLI_Foundation_Model/blob/main/2_Pre_train_Foundation_model/Pre_train_Foundation_model.ipynb)|
|best_resnet_model_finetuned_XX%.pth  | Fine-tuned foundation model, here XX means training dataset is XX fraction of total MEGAPOLI data (XX *2 fraction of fine-tuning training dataset) | [Fine-tune_different_data_size.ipynb](https://github.com/envdes/code_MEGAPOLI_Foundation_Model/blob/main/3_Fine_tune_different%20_data_size/Fine-tune_different_data_size.ipynb)  | 
|best_resnet_model_finetuned_XX%_test.pth  | Fine-tuned foundation model ('Swapped Temporal Order' experiment), here XX means training dataset is XX fraction of total MEGAPOLI data (XX *2 fraction of fine-tuning testing dataset), model performance evaluated on fine-tuning training dataset | [Fine-tune_different_data_size.ipynb](https://github.com/envdes/code_MEGAPOLI_Foundation_Model/blob/main/3_Fine_tune_different%20_data_size/Fine-tune_different_data_size.ipynb)  | 
|best_resnet_model_finetuned_50%_XX_xxxx.csv  | Fine-tuned foundation model, here XX means input feature size and xxxx means input feature subset combination | [Fine-tune_different_input_size.ipynb](https://github.com/envdes/code_MEGAPOLI_Foundation_Model/blob/main/4_Fine_tune_different_input_feature_size/Fine-tune_different_input_size.ipynb)  | 
|AutoML_XX%Data.csv  | Model trained and selected by AutoML, here XX means training dataset is XX fraction of total MEGAPOLI data (XX *2 fraction of fine-tuning training dataset) | [AutoML.ipynb](https://github.com/envdes/code_MEGAPOLI_Foundation_Model/blob/main/3_Fine_tune_different%20_data_size/AutoML.ipynb)  | 
|AutoML_XX%Data_test.csv  | Model trained and selected by AutoML ('Swapped Temporal Order' experiment), here XX means training dataset is XX fraction of total MEGAPOLI data (XX *2 fraction of fine-tuning testing dataset), model performance evaluated on fine-tuning training dataset | [AutoML.ipynb](https://github.com/envdes/code_MEGAPOLI_Foundation_Model/blob/main/3_Fine_tune_different%20_data_size/AutoML.ipynb)  |

## Acknowledgments
- This work made use of the facilities of the N8 Centre of Excellence in Computationally Intensive Research (N8 CIR) provided and funded by the N8 research partnership and EPSRC (Grant No. EP/T022167/1). The Centre is coordinated by the Universities of Durham, Manchester and York.
- The authors acknowledge the assistance given by Research IT and Computational Shared Facility 3 (CSF3) at The University of Manchester.
- Z.Z. appreciates the support provided by the academic start-up funds from the Department of Earth and Environmental Sciences at The University of Manchester and the Aerosol Science Career Development Grant from The Aerosol Society. The authors declare no conflict of interest.