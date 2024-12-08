# ECE1512_ProjectB_PartB: Efficiency Optimization Using Dynamic Token Pruning

## Overview
This repository contains the implementation and results of our project for ECE1512, where we focus on improving transformer model efficiency using **Dynamic Token Pruning**. By dynamically reducing the number of tokens processed in each layer, this method significantly decreases FLOPs and execution time while maintaining the model's accuracy.

## Features
- **Dynamic Token Pruning**:
  - Computes saliency scores to identify unimportant tokens.
  - Prunes tokens dynamically at each transformer layer.
- **Efficiency Profiling**:
  - Detailed measurement of FLOPs and CUDA memory usage for baseline and pruned models.
- **Accuracy Evaluation**:
  - Ensures no accuracy degradation when pruning tokens.

## Results
- **Baseline Model**:
  - FLOPs: **12.908G**
  - CUDA Time: **12.490ms**
  - Accuracy: **52.5%**
- **Pruned Model**:
  - FLOPs: **8.874G**
  - CUDA Time: **9.968ms**
  - Accuracy: **52.5%**
  
Achieved a **31.2% FLOPs reduction** and a **20.2% runtime reduction** while maintaining the same accuracy.

## Repository Structure
```
src/
├── model.py               # Transformer models (baseline and pruned)
├── pruning.py             # Implementation of dynamic token pruning
├── utils.py               # Helper functions for profiling and evaluation
experiments/
├── profiling.ipynb        # FLOPs and memory profiling notebook
├── evaluation.ipynb       # Accuracy evaluation scripts
results/
├── profiling_baseline.txt # Baseline profiling results
├── profiling_pruned.txt   # Pruned model profiling results
README.md                  # Project documentation
```

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA-enabled GPU

### Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/ADglory/ECE1512_2024F_ProjectB_PartB_Repo_YingshunLu_Minghao-Ma
cd ECE1512_2024F_ProjectB_PartB_Repo_YingshunLu_Minghao-Ma
pip install -r requirements.txt
```

### Usage

#### Run Profiling
To measure FLOPs and CUDA memory usage:
```bash
python src/profiling.py
```

#### Evaluate Model Accuracy
To evaluate the accuracy of baseline and pruned models:
```bash
python src/evaluate.py
```

## Discussion
The implementation successfully demonstrates the efficiency of dynamic token pruning, achieving:
- **31.2% FLOPs reduction**
- **20.2% runtime reduction**
- No degradation in accuracy (maintained at **52.5%**).

This approach highlights the feasibility of token pruning in transformer models for practical deployment.

## Authors
- **Yingshun Lu**
- **Minghao Ma**


## Acknowledgements
This project is part of ECE1512 coursework at the University of Toronto. The focus is on transformer efficiency optimization using innovative token pruning techniques.
