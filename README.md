# ID_of_AI_Images
UVA DS6050 Fall 2025 Group 7 Final Project. Identification of AI Images. 

Currently using conda to manage environments

# ***Using Random Seed of 6050 for any random process***

```conda create -n id_ai```

```conda install python=3.12```

```pip install -r requirements.txt```

# Getting AI-GenBench to run:

GitBash: ```git clone https://github.com/MI-BioLab/AI-GenBench.git```

```cd AI-GenBench```

Anaconda Prompt: ```conda create -n ai_genbench```

```conda activate ai_genbench```

```conda install python=3.12```

*make sure conda prompt terminal is cd'd into AI-GenBench

```pip install -r requirements_dataset_creation.txt```

---
*Jimmy 11/4

Attempting to collect all data from AI_GenBench Using there 'Simple dataset creation steps':
https://github.com/MI-BioLab/AI-GenBench/blob/main/dataset_creation/README.md#simple-dataset-creation-steps

- For ImageNet, had to create an account on their website with edu email to gain access to dataset. (AI-GenBench only uses 2012). Downloads in recursive .tar files, and the AI-GenBench website does not clarify what to do with this. 
- For RAISE, downloaded the filelist of their website and executed AI-GenBench's download script, failed multiple times and now I cannot access the website. 