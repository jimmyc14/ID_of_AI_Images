# ID_of_AI_Images
UVA DS6050 Fall 2025 Group 7 Final Project. Identification of AI Images. 

This project incorperates the [AI-GenBench](https://github.com/MI-BioLab/AI-GenBench) dataset and workflow for the use of additional models for testing and evaluation. A 40% sample was used from the full AI-GenBench dataset, and used to train and validate our models. The AI-GenBench workflow uses a windowed training approach, where a model is trained on temporal windows of AI generated images. This process allows for models to be evaluated on their ability to generalize for future models. 

We propose 2 initial models, a simple ResNet50 and ViT model, with the addition of FFT (Fast Fourier Transform) versions of each model. This results in 4 total models. 

We also recreate the AI-GenBench temporal training workflow, as well as a general 'normal' workflow that uses all images and does not consider temporal information. 

These training methods are split into 2 groups, temporal and normal. We focus on temporal training, as this reflects the AI-GenBench's original method, as well as allowing for the group to see a model's ability to generalize. However 'normal' training is still available. 

2 Methods of training are available, via a juptyer notebook, and via the commandline. 

## Contents

- [Setting Up Environment](#setting-up-our-environment)
- [Getting the Dataset on Your Machine](#getting-the-dataset-on-your-machine)
- [Training Models](#training-models)
    - [Available Models](#available-models)
    - [Main Parameters](#main-parameters)
    - [Via Jupyter Notebook](#jupyter-notebook-training)
    - [Via Commandline](#commandline-training)
- [Viewing Results](#viewing-results)
- [Running Inference](#running-inference)
- [Dataset Stats](dataset_stats.md)

Extra
- [Getting GitHub Sync'd with Rivanna (if GitHub editor)](#connecting-your-rivanna-session-to-the-main-github-repo)
- [Getting AI-GenBench Running (if attempting to recreate original dataset)](#getting-ai-genbench-to-run-only-needed-if-trying-to-compile-the-data-the-way-they-do)
---

# Setting up our environment

Currently using conda to manage environments

Clone this repo: ```https://github.com/jimmyc14/ID_of_AI_Images.git```

cd into directory: ```cd ID_of_AI_Images```

For Rivanna: Run this first to load Conda ```module load miniforge/24.11.3-py3.12```

Create conda environment: ```conda create -n id_ai```

Make sure to activate environment: ```conda activate ai_id```

Install python: ```conda install python=3.12```

Install requirements for project: ```pip install -r requirements.txt```

FOR RIVANNA: Run the following line to make sure JupyterLab can see our new environment ```python -m ipykernel install --name id_ai --display-name id_ai --prefix ~/.local```. You may need to close the tab and reopen the Rivanna session to make it show up as a kernal option within a notebook. 

--- 
## Getting the dataset on your machine. 

### Recommend Using a CPU session in Rivanna with at least 8 cores to make the file extraction faster. 

- Due to the size of the AI-GenBench compiled dataset (~100gb), we have uploaded a 40% version to huggingface.

- IMPORTANT FOR RIVANNA: in the terminal run ```module load git-lfs``` and ```git lfs install```, this loads gits large file storage which allows us to get the data

- Use ```git clone https://huggingface.co/datasets/szp2fv/DS6050_Ai_Detection``` to copy the dataset. This will take some time as the dataset is ~40 gb. https://huggingface.co/datasets/szp2fv/DS6050_Ai_Detection

- ```cd DS6050_Ai_Detection```
- ```rm -rf .git``` this is recommended to save space 

- The data are stored in arrows, so you will have to extract them using the following script from the repo:
[https://github.com/jimmyc14/ID_of_AI_Images/blob/main/data_download_management/parallel_test.ipynb](https://github.com/jimmyc14/ID_of_AI_Images/blob/main/data_download_management/parallel_test.ipynb) this requires creating the conda environment as stated above. 

- Fill in the path you just clone the repo to in the 'parallel_test.ipynb' script. For example mine is: ```C:/Users/Jimmy/OneDrive/Desktop/test/DS6050_Ai_Detection``` as seen in the current script. FOR RIVANNA: make sure to include the directory structure you are using, for example with SCRATCH: ```/scratch/{uva_id}/path/to/DS6050_Ai_Detection```

- As the script runs, it will copy all images into a respective temp_ folder, then delete the original folder storing the arrows, to save some space. It will then rename the temp folders.

    - note: This will also take some time, it took roughly ~30-60 minutes for me. 
    - note2: Roughly ~300 images in the dataset are EXTREMELY large, causing issues with saving. They are therefore resized 10x smaller to avoid errors during this extraction process. These images were 30720x20562 pixels originally. 

- Due to this, if a failure occurs mid-extraction, make sure all 4 folders containing arrows are there before attempting to run again. If not, you may need to clone the dataset (or at least the missing folders) again.

- If you are having issues, please see a [more in depth data download documentation](https://myuva-my.sharepoint.com/:w:/g/personal/szp2fv_virginia_edu/IQCxSKjQNzvKR4r8v_2MjzGqAaWmcJ2-rnDAPecnqIpWgfc?e=IqS5UA)
---

# Training Models

Training models will generate logging data in the folder 'logs'. This folder will automatically be created when you train the first model. For each run, there will be a .txt and .json generated that will be updated at each step/epoch. A .pth file that stores all the best weights for the model will also be generated as the model is trained and store in the training folder. This .pth file will be updated when a new best F1 score is made. 

A dataset split .csv will also be generated in the log file upon each creation of a new dataset within the traning code. 

### Available Models
4 models are available to train. 
- [ResNet50: 'resnet50'](models/resnet.py)
- [ResNet50+FFT: 'resnet50_fft'](models/resnet_fft.py)
- [ViT: 'vit'](models/vit.py)
- [ViT+FFT: 'vit_fft'](models/vit_fft.py)

### Main Parameters
The main parameters as found within each notebook / for command line running
- data_root: Str, The main folder where all the data is located for training/validation. 
- model_name: Str. The model you want to used for training, 1 of the 4 available models.
- train_percent: Float. Percentage of all data available to train model on. Must be between 0.1-1.0. Helpful for testing. 
- num_epochs: Int. Number of epochs to train the model. For temporal model, epochs will be run back to back for each window step.
- batch_size: Int. Number of batches to train at a time. Must be divisable by 8. 
- learning_rate: Float. The learning rate to use for model training.
- num_workers: Int. Number of workers to use for data loading. (Bugged, set to 0 for the moment)
- jpeg_comp: Boolean. If True, will augment training data with jpeg compression, otherwise no compression. 
- save_model_name: Str. Optional, will add str at end of model name for logging. 

## Jupyter Notebook Training

For both Temporal and Normal Models, training via notebooks can be done from the [normal training](training/training_normal_nb.ipynb) and [temporal training](training/training_temporal_nb.ipynb) notebooks. 

In these notebooks, simply update the configuration block with your data path and your desired training parameters.

## Commandline Training

For commandline training, we could only get it working through a conda commandline, but we are sure some simple work can be done to run on normal commandline. 

- cd into the training directory within ID_of_AI_Images: ```cd training```
- For Temporal training: run the cmmd_line_temporal.py. 
    - example usage: ```python cmmd_line_temporal.py --data_root "C:/Users/Jimmy/OneDrive/Desktop/test/DS6050_Ai_Detection" --model_name "resnet50" 
--batch_size 16 --num_epochs 1 --learning_rate 1e-4 --train_percent 0.1 --num_workers=0 --jpeg_comp True --save_model_name "cmmd_line" ```

- For Normal training: run the cmmd_line_normal.py.
    - example usage: ```python cmmd_line_normal.py --data_root "C:/Users/Jimmy/OneDrive/Desktop/test/DS6050_Ai_Detection" --model_name "resnet50" 
--batch_size 16 --num_epochs 1 --learning_rate 1e-4 --train_percent 0.1 --num_workers=0 --jpeg_comp True --save_model_name "cmmd_line"
''' ```

# Viewing Results

There are 2 json parsers available in this project.
- One for [temporal training results](temporal_json_parsing.ipynb)
- One for [normal training results](regular_json_parsing.ipynb)

Within each notebook, simply add the path to your jsons to the json_path list to compare them to each other. Some sample jsons are available within the 'training/logs' folder. 

# Running Inference

A notebook can be found to run some inference our your own images. In the 'inference' folder, the [inference notebook](inference/inference.ipynb) can be used to run your own trained model. Using a dictionary of each 4 base models, you can give the paths to every .pth weight files available to test. The folder 'images' within the 'inference' folder is where you can upload images for your models to test. 

This will generate a .txt file with results as well.

4 example images and 4 model .pth weights are uploaded for you to test. 

# Connecting Your Rivanna session to the main github repo

- First, must accept invation to be a collaborator for the github repo.

- In Rivanna: ```ssh-keygen -t ed25519 -C "{your UVA email}"```

- ```cat ~/.ssh/id_ed25519.pub``` -> This will show your ssh-key for your rivanna.

- Now go to GitHub, and add that key to your GitHub -> SSH and GPG keys: https://github.com/settings/keys. Click on 'New SSH key' and add the key from Rivanna. This will link your github account to your Rivanna session

- Back in Rivanna, navigate to the ID_of_AI_Images folder, and do ```git remote set-url origin git@github.com:jimmyc14/ID_of_AI_Images.git```. This will ensure your cloned session uses ssh.

- You should now be able to fully use git to manage the project!

---
# Getting AI-GenBench to run (only needed if trying to compile the data the way they do):
## No need to do this for the current project workflow

GitBash: ```git clone https://github.com/MI-BioLab/AI-GenBench.git```

```cd AI-GenBench```

Anaconda Prompt: ```conda create -n ai_genbench```

```conda activate ai_genbench```

```conda install python=3.12```

*make sure conda prompt terminal is cd'd into AI-GenBench

```pip install -r requirements_dataset_creation.txt```

---
## Compiling the Ai-GenBench Dataset

Attempting to collect all data from AI_GenBench Using there 'Simple dataset creation steps':
https://github.com/MI-BioLab/AI-GenBench/blob/main/dataset_creation/README.md#simple-dataset-creation-steps

- Due to access and storage issues, for the 'real' images in the AI-GenBench dataset, only the LAION-400m and COCO datasets were used. ImageNet and Raise were excluded. 

- AI-GenBench's fake dataset (hosted on huggingface by the authors) was used. 

- More information about the exact distribution of data is coming, but AI-GenBench originally had 360k total images. There were 180k fake images, and 180k real images, with a training/validation split of 80/20, so 144k training, and 36k validation per image type of real and fake. 

- In gathering the real training images, the LAION dataset contained 134,453 images, and the coco dataset contained 123,287 images. They were randomly subsampled equally to be combined for a total of 180,000 images. 

- Since only the LAION and COCO image datasets were used, the amount of available real validation images was 35,426 images. The fake validation images were randomly subsampled to match the amount of real images available, so 574 images were removed. 

- As stated above, we are using a random 40% subset of the full 360,000 proposed images, for a total of 143,450 images. 

--- 
