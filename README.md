# ID_of_AI_Images
UVA DS6050 Fall 2025 Group 7 Final Project. Identification of AI Images. 

Currently using conda to manage environments

### ***Using Random Seed of 6050 for any random process***

---

# Setting up our environment

Clone this repo: ```https://github.com/jimmyc14/ID_of_AI_Images.git```

cd into directory: ```cd ID_of_AI_Images```

For Rivanna: Run this first to load Conda ```module load miniforge/24.11.3-py3.12```

Create conda environment: ```conda create -n id_ai```

Make sure to activate environment: ```conda activate ai_id```

Install python: ```conda install python=3.12```

Install requirements for project: ```pip install -r requirements.txt```

FOR RIVANNA: Run the following line to make sure JupyterLab can see our new environment ```python -m ipykernel install --name id_ai --display-name id_ai --prefix ~/.local```. You made need to close the tab and reopen the Rivanna session to make it show up as a kernal option within a notebook. 

--- 
## Getting the dataset on your machine. 

### Recommend Using a CPU session in Rivanna with at least 8 cores to make the file extraction faster. 

- Due to the size of the AI-GenBench compiled dataset (~100gb), we have uploaded a 40% version to huggingface.

- IMPORTANT FOR RIVANNA: in the terminal run ```module load git-lfs``` and ```git lfs install```, this loads gits large file storage which allows us to get the data

- Use ```git clone https://huggingface.co/datasets/szp2fv/DS6050_Ai_Detection``` to copy the dataset. This will take some time as the dataset is ~40 gb. https://huggingface.co/datasets/szp2fv/DS6050_Ai_Detection
* - *note there may be a hidden .git file in the dataset that is large after cloning, feel free to delete if needed. 

- The data are stored in arrows, so you will have to extract them using the following script from the repo:
[https://github.com/jimmyc14/ID_of_AI_Images/blob/main/data_download_management/get_huggingface_data.py](https://github.com/jimmyc14/ID_of_AI_Images/blob/main/data_download_management/parallel_test.ipynb)

- Fill in the path you just clone the repo to in the 'parallel_test.ipynb' script. For example mine is: ```C:/Users/Jimmy/OneDrive/Desktop/test/DS6050_Ai_Detection``` as seen in the current script. FOR RIVANNA: make sure to include the directory structure you are using, for example with SCRATCH: ```/scratch/{uva_id}/path/to/DS6050_Ai_Detection```

- As the script runs, it will copy all images into a respective temp_ folder, then delete the original folder storing the arrows, to save some space. It will then rename the temp folders.

* - note: This will also take some time, it took roughly ~30-60 minutes for me. 
* - note2: Roughly ~300 images in the dataset are EXTREMELY large, causing issues with saving. They are therefore resized 10x smaller to avoid errors during this extraction process. These images were 30720x20562 pixels originally. 

- Due to this, if a failure occurs mid-extraction, make sure all 4 folders containing arrows are there before attempting to run again. If not, you may need to clone the dataset (or at least the missing folders) again.

---
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

- Since only the LAION and COCO image datasets were used, the amount of available real validation images was 35,426 images. The fake validation images were randomly subsampled to match the amount of real images available, so 574 images were removed. 

---wip---

### * need to check the stats on these * TODO: Develop a table of image counts by data source for train/validation, real/fake. Can be done with the csvs quite easily (they track the source)
- In gathering the real training images, the LAION dataset contained 134,453 images, and the coco dataset contained 123,287 images.   
