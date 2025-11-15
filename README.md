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

FOR RIVANNA: Run the following line to make sure JupyterLab can see our new environment ```python -m ipykernel install --name id_ai --display-name id_ai --prefix ~/.local```. You may need to close the tab and reopen the Rivanna session to make it show up as a kernal option within a notebook. 

--- 
## Getting the dataset on your machine. 

### Recommend Using a CPU session in Rivanna with at least 8 cores to make the file extraction faster. 

- Due to the size of the AI-GenBench compiled dataset (~100gb), we have uploaded a 40% version to huggingface.

- IMPORTANT FOR RIVANNA: in the terminal run ```module load git-lfs``` and ```git lfs install```, this loads gits large file storage which allows us to get the data

- Use ```git clone https://huggingface.co/datasets/szp2fv/DS6050_Ai_Detection``` to copy the dataset. This will take some time as the dataset is ~40 gb. https://huggingface.co/datasets/szp2fv/DS6050_Ai_Detection
* - *note there may be a hidden .git file in the dataset that is large after cloning, feel free to delete if needed. 

- The data are stored in arrows, so you will have to extract them using the following script from the repo:
[https://github.com/jimmyc14/ID_of_AI_Images/blob/main/data_download_management/parallel_test.ipynb](https://github.com/jimmyc14/ID_of_AI_Images/blob/main/data_download_management/parallel_test.ipynb)

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

- In gathering the real training images, the LAION dataset contained 134,453 images, and the coco dataset contained 123,287 images. They were randomly subsampled equally to be combined for a total of 180,000 images. 

- Since only the LAION and COCO image datasets were used, the amount of available real validation images was 35,426 images. The fake validation images were randomly subsampled to match the amount of real images available, so 574 images were removed. 

- As stated above, we are using a random 40% subset of the full 360,000 proposed images, for a total of 143,450 images. 

--- 
# Dataset Stats

## Split Counts

| Split | Count   |
|-------|--------:|
| Training | 115,200 |
| Validation | 28,340 |
|────────────|
| Total | 143,540 |

## Image Type Counts

| Type | Count   |
|-------|--------:|
| Real | 71,770 |
| Fake | 71,770 |
|──────|
| Total | 143,540 |

## Image Type by Split Counts

| Split/Type | Count   |
|-------|--------:|
| Real Train | 57,600 |
| Fake Train | 57,600 |
| Real Validation | 14,170 |
| Fake Validation | 14,170 |
|─────────────────|
| Total | 143,540 |

## Real Image Dataset Sources

### Total Real from Souces

| Real Source | Count   |
|-------|--------:|
| Laion | 41,043 |
| Coco | 30,727 |
|──────|
| Total | 71,770 |

### Real Split Sources

| Real Split Source | Count   |
|-------|--------:|
| Train Laion | 28,825 |
| Train Coco | 28,775 |
| Validation Laion | 12,218 |
| Validation Coco | 1,952 |
|─────────────────|
| Total | 71,770 |

* note the low number of coco validation images. This is due to the original coco validation dataset only having 5,000 images

## Fake Image Dataset Sources

The Fake Images are sourced from 12 datasets, and 36 total generators. Some of the datasets have different splits listed, and some of the generators have different releases listed. 

### Fake Generator Counts

| Generator                     | Count   |
|-------------------------------|--------:|
| ADM                           |   1,976 |
| BigGAN                        |   1,976 |
| CIPS                          |   2,018 |
| Cascaded Refinement Networks  |   2.008 |
| CycleGAN                      |   1,953 |
| DALL-E 3                      |   2,086 |
| DDPM                          |   2,033 |
| DeepFloyd IF                  |   2,008 |
| Denoising Diffusion GAN       |   2,033 |
| Diffusion GAN (ProjectedGAN)  |   2,024 |
| Diffusion GAN (StyleGAN2)     |   1,942 |
| FLUX 1 Dev                    |   1,973 |
| FLUX 1 Schnell                |   2,026 |
| FaceSynthetics                |   1,960 |
| GANformer                     |   1,935 |
| GauGAN                        |   2,005 |
| Glide                         |   1,927 |
| IMLE                          |   1,978 |
| LaMa                          |   1,999 |
| Latent Diffusion              |   2,025 |
| MAT                           |   2,024 |
| Midjourney                    |   1,939 |
| Palette                       |   1,912 |
| ProGAN                        |   1,950 |
| ProjectedGAN                  |   1,997 |
| SN-PatchGAN                   |   2,043 |
| Stable Diffusion 1.4          |   1,994 |
| Stable Diffusion 1.5          |   1,982 |
| Stable Diffusion 2.1          |   1,990 |
| Stable Diffusion XL 1.0       |   1,961 |
| StarGAN                       |   1,998 |
| StyleGAN1                     |   2,062 |
| StyleGAN2                     |   1,969 |
| StyleGAN3                     |   2,025 |
| VQ-Diffusion                  |   2,025 |
| VQGAN                         |   2,019 |
|───────────────────────────────|
| Total | 71,770 |

### Fake Image Origin Dataset Counts

| Origin Dataset             | Count   |
|---------------------------|--------:|
| Aeroblade                 |     399 |
| Artifact                  |  31,881 |
| DDMD                      |   6,343 |
| DMimageDetection/test     |   2,706 |
| DMimageDetection/train    |     187 |
| DMimageDetection/valid    |     208 |
| DRCT                      |   2,210 |
| ELSA_D3/train             |   2,269 |
| ELSA_D3/valid             |   2,289 |
| Forensynths/test          |   8,900 |
| Forensynths/train         |     113 |
| Forensynths/valid         |     113 |
| GenImage/train            |   5,481 |
| GenImage/val              |     210 |
| Imageinet                 |   2,464 |
| Polardiffshield           |     785 |
| SFHQ-T2I                  |   4,403 |
| Synthbuster               |     809 |
|───────────────────────────|
| Total | 71,770 |

## Fake Training Split Generators and Sources

| Generator                    | Count |
| ---------------------------- | ----: |
| DALL-E 3                     |  1680 |
| CIPS                         |  1664 |
| SN-PatchGAN                  |  1644 |
| StyleGAN1                    |  1637 |
| Latent Diffusion             |  1635 |
| MAT                          |  1634 |
| Diffusion GAN (ProjectedGAN) |  1630 |
| FLUX 1 Schnell               |  1628 |
| StyleGAN3                    |  1623 |
| DeepFloyd IF                 |  1621 |
| DDPM                         |  1619 |
| VQ-Diffusion                 |  1618 |
| StarGAN                      |  1616 |
| GauGAN                       |  1612 |
| Cascaded Refinement Networks |  1609 |
| StyleGAN2                    |  1606 |
| ProjectedGAN                 |  1603 |
| ADM                          |  1602 |
| Stable Diffusion 2.1         |  1600 |
| Denoising Diffusion GAN      |  1597 |
| VQGAN                        |  1594 |
| BigGAN                       |  1590 |
| Stable Diffusion 1.4         |  1587 |
| FLUX 1 Dev                   |  1586 |
| LaMa                         |  1584 |
| IMLE                         |  1584 |
| Stable Diffusion 1.5         |  1579 |
| CycleGAN                     |  1575 |
| GANformer                    |  1566 |
| Stable Diffusion XL 1.0      |  1565 |
| FaceSynthetics               |  1564 |
| ProGAN                       |  1564 |
| Glide                        |  1561 |
| Diffusion GAN (StyleGAN2)    |  1561 |
| Midjourney                   |  1545 |
| Palette                      |  1517 |
|──────────────────────────────|
| Total | 57,600 |

| Origin Dataset         | Count |
| ---------------------- | ----: |
| Artifact               | 25577 |
| Forensynths/test       |  7160 |
| DDMD                   |  5073 |
| GenImage/train         |  4400 |
| SFHQ-T2I               |  3530 |
| DMimageDetection/test  |  2205 |
| Imaginet               |  1968 |
| ELSA_D3/valid          |  1845 |
| ELSA_D3/train          |  1822 |
| DRCT                   |  1765 |
| Synthbuster            |   645 |
| Polardiffshield        |   629 |
| Aeroblade              |   315 |
| DMimageDetection/valid |   171 |
| GenImage/val           |   167 |
| DMimageDetection/train |   144 |
| Forensynths/train      |    96 |
| Forensynths/valid      |    88 |
|────────────────────────|
| Total | 57,600 |

## Fake Validation Split Generators and Sources

| Generator                    | Count |
| ---------------------------- | ----: |
| Denoising Diffusion GAN      |   436 |
| StyleGAN1                    |   425 |
| VQGAN                        |   425 |
| LaMa                         |   415 |
| DDPM                         |   414 |
| Stable Diffusion 1.4         |   407 |
| DALL-E 3                     |   406 |
| Stable Diffusion 1.5         |   403 |
| VQ-Diffusion                 |   402 |
| StyleGAN3                    |   402 |
| Cascaded Refinement Networks |   399 |
| SN-PatchGAN                  |   399 |
| FLUX 1 Schnell               |   398 |
| Stable Diffusion XL 1.0      |   396 |
| FaceSynthetics               |   396 |
| Palette                      |   395 |
| Midjourney                   |   394 |
| IMLE                         |   394 |
| Diffusion GAN (ProjectedGAN) |   394 |
| ProjectedGAN                 |   394 |
| GauGAN                       |   393 |
| Stable Diffusion 2.1         |   390 |
| MAT                          |   390 |
| Latent Diffusion             |   390 |
| DeepFloyd IF                 |   387 |
| FLUX 1 Dev                   |   387 |
| BigGAN                       |   386 |
| ProGAN                       |   386 |
| StarGAN                      |   382 |
| Diffusion GAN (StyleGAN2)    |   381 |
| CycleGAN                     |   378 |
| ADM                          |   374 |
| GANformer                    |   369 |
| Glide                        |   366 |
| StyleGAN2                    |   363 |
| CIPS                         |   354 |
|──────────────────────────────|
| Total | 14,170 |

| Origin Dataset         | Count |
| ---------------------- | ----: |
| Artifact               |  6304 |
| Forensynths/test       |  1740 |
| DDMD                   |  1270 |
| GenImage/train         |  1081 |
| SFHQ-T2I               |   873 |
| DMimageDetection/test  |   501 |
| Imaginet               |   496 |
| ELSA_D3/train          |   447 |
| DRCT                   |   445 |
| ELSA_D3/valid          |   444 |
| Synthbuster            |   164 |
| Polardiffshield        |   156 |
| Aeroblade              |    84 |
| GenImage/val           |    43 |
| DMimageDetection/train |    43 |
| DMimageDetection/valid |    37 |
| Forensynths/valid      |    25 |
| Forensynths/train      |    17 |
|────────────────────────|
| Total | 14,170 |
