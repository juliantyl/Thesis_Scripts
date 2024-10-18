# Optimizing Code Classification and Clone Detection: Investigating Semantic and Structural Representations in Machine Learning Models - Undergraduate Thesis Project

Welcome to the scripts for my undergraduate thesis project: **Optimizing Code Classification and Clone Detection: Investigating Semantic and Structural Representations in Machine Learning Models**.

## Project Structure

This repository contains two models, which can be found in their respective folders:

- **Token Based Classification**: [Token based representation/classification.py]
- **Token Based Clone Detection**: [Token based representation/clone_detection.py]
- **AST Based Classification**: [AST based representation/AST_classification.py]
- **AST Based Clone Detection**: [AST based representation/AST_clone_detection.py]
- **CFG Based Classification**: [CFG based representation/graph_classification.py]
- **CFG Based Clone Detection**: [CFG based representation/graph_clone_detection.pye]

## Required Datasets

In order to run the models, two datasets are required. These datasets are too large to upload to GitHub, so you'll need to download them separately:

1. **Clone Detection Dataset**: 
   - Available at: [BigCloneBench](https://github.com/clonebench/BigCloneBench)

2. **Classification Dataset**: 
   - Available from the **Online Judge Dataset**, made public by Mou, L., et al. in the paper:  
     *"Convolutional Neural Networks over Tree Structures for Programming Language Processing"*

## Setup Instructions

To run the project, ensure that you create the necessary environment using the provided `environment.yml` file. You can do this using `conda`:

```bash
conda env create -f environment.yml
conda activate [your-env-name]
```


