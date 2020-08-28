# Udacity_Customer-Segmentation
Analysis of demographics data for customers of a mail-order sales company in Germany, incl. the comparison of it against demographics information for the general population. The goal was to identify the parts of the population that best describe the core customer base of the company. The model was then applied on a third dataset with demographics information to predict the response rate of future possible customers.

## Table of Contents
1. [ Installation. ](#insta)
2. [ File Description and Folder Structure. ](#desc)
3. [ Results. ](#res)

<a name="insta"></a>
## 1. Installation

The main analysis was conducted with a Jupyter Notebook in a conda environment. The Kernel was Python 3 (Version 3.6). 

The main libraries used were:

# import libraries here; add more as necessary
* Numpy (1.19.1)
* threadpoolctl (2.1.0)
* SKlearn (0.23.2)
* imbalanced-learn (0.7.0)
* graphviz (0.14.1)
* XGBoost (1.2.0)
* Pandas
* Matplotlib
* Seaborn
* Pickle

Anaconda should provide essential libraries. XGBoost, graphviz, imbalanced has to be installed via !pip. SKlearn may have to be updated to the latest version. Therefore, you have to run the following command:

```
!pip install scikit-learn==0.23.2
```

<a name="desc"></a>
## 2. File Description and Folder Structure

```
.
├── 01_proposal
│   └── capstone_proposal.md -----------------------------# PROVIDES THE INITIAL PROPOSAL ABOUT THE ANALYSIS, STEPS, ALGORITHM USED,....
├── 02_images --------------------------------------------# PROVIDES THE IMAGES USED IN THE PROPOSAL AND IN THE PROJECT REPORT
│   ├── Customer_Distribution.PNG
│   ├── PCA_analysis.png
│   ├── XGBoost_trained.png
│   └── ....
├── 03_dataset information
│   ├── DIAS Attributes - Values 2017.xlsx ---------------# PROVIDES INFORMATION ON EACH COLUMN IN THE DATASET INCL. GIVEN ATTRIBUTES 
│   └── DIAS Information Levels - Attributes 2017.xlsx ---# PROVIDES A HIGH-LEVEL CLUSTERING ACROSS ALL >300 COLUMNS IN THE DATASETS
├── README.md --------------------------------------------# README FILE PROVIDING GENERAL INFORMATION ON REPOSITORY STRUCTURE AND INSTALLATION REQUIREMENTS
├── helper_classification.py -----------------------------# HELPER FUNCTIONS TO PROCESS, VISUALIZE AND TRAIN SUPERVISED LEARNING MODEL AND CONNECTED DATASETS
├── helper_segmentation.py -------------------------------# HELPER FUNCTIONS TO PROCESS, VISUALIZE AND TRAIN UNSUPERVISED LEARNING MODEL AND CONNECTED DATASETS
└── project_report.md ------------------------------------# DETAILED DOCUMENTATION OF ANALYSIS RESULTS INCL. VISUALIZATIONS AND INTERPRETATION 
```

<a name="res"></a>
## 3. Results
The main findings of the conducted analysis and the respective written code can be found in project_report.md.
