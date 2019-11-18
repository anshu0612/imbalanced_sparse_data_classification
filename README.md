## CS5228 Knowledge Discovery and Data Mining Final Project on Time Series Data Binary Classification 
[Kaggle Competition](https://www.kaggle.com/c/1910-cs5228-knowledge-discovery-and-data-mining/)

The problem given was that of time series binary data classification. The solution to this problem involves careful feature engineering and data preprocessing, boosting and deep learning models and ensembling of these models. 

The solution achieved an AUC score of 93.461% on [Kaggle's private leadership board](https://www.kaggle.com/c/1910-cs5228-knowledge-discovery-and-data-mining/leaderboard)

# Prerequisites
- Python 3.6 or above
- Virtualenv
- Kaggle API key set up (https://github.com/Kaggle/kaggle-api)

# Directory Structure
|- src
|   |- ml_feat.py   : Code for LightGBM based approach
|   |- nn.py        : Code for Neural Network based training
|   |- ensemble.py  : Code to combine outputs of both
|   |- analyze.ipynb: Basic data analysis
|- requirements.txt : Python requirements file
|- README.md        : This file
|- outputs          : Folder containing generated CSV files

## Installation, Training and Generating Predictions Instructions
- Run run.sh script on your machine.
- The final labels file will be generated as output.csv in the outputs directory

## Downloading dataset from Kaggle
The easiest way to interact with Kaggle’s dataset is via Kaggle Command-line tool (CLI). Below are the steps to setup Kaggle CLI and use it to download the dataset

#### The Setup 
1. Install the Kaggle CLI
To get started to Kaggle CLI we will need Python, open terminal and type command ``pip install kaggle``
2. API Credentials
Once we have Kaggle installed, type kaggle to check it is installed and we will get an output similar to this

![IMAGE](https://nndl.s3.amazonaws.com/1.png)

In the above line, we will see the path (highlighted) of where to put your kaggle.json file.
To get kaggle.json file go to:
https://www.kaggle.com/<username>/account

In the API section, click Create New API Token. And copy it the path mentioned in the terminal output.

![IMAGE](https://nndl.s3.amazonaws.com/2.png)

Type kaggle once again to check.
![IMAGE](https://nndl.s3.amazonaws.com/3.png)

In some case, even after copying the credentials will not work even though the file is placed in the correct location due incorrect permission. Just type the exact command and it will start working

#### Downloading Dataset via CLI

We can open kaggle help via `kaggle -h`
For getting info on competitions we can type `kaggle competitions download -h`
whatever the Kaggle CLI command is, add -h to get help.

#### Download Entire Dataset
To download the dataset, go to Data subtab on the competition page. In API section we will find the exact command that we can copy to the terminal to download the entire dataset.

![IMAGE](https://nndl.s3.amazonaws.com/4.png)

The syntax is like `kaggle competitions download <competition name>`
One the dataset is downloaded extract the dataset and use it.
