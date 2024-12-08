ECE1512: Efficiency Optimization Using Dynamic Token Pruning
Overview
This repository contains the implementation and results of our project for ECE1512, focusing on optimizing transformer model efficiency using Dynamic Token Pruning. The method dynamically reduces the number of tokens processed in the transformer layers, leading to significant FLOPs and time reductions while maintaining accuracy.

Features
Dynamic Token Pruning:
Computes saliency scores to identify unimportant tokens.
Dynamically prunes tokens across transformer layers.
Efficiency Profiling:
Detailed FLOPs and CUDA memory profiling for baseline and pruned models.
Accuracy Evaluation:
Maintains accuracy on toy binary classification datasets.
Results
Baseline Model:

FLOPs: 12.908G
CUDA Time: 12.490ms
Accuracy: 52.5%
Pruned Model:

FLOPs: 8.874G
CUDA Time: 9.968ms
Accuracy: 52.5%
Repository Structure


├── src/
│   ├── model.py               # Transformer models (baseline and pruned)
│   ├── pruning.py             # Dynamic token pruning logic
│   ├── utils.py               # Utility functions for profiling and analysis
├── experiments/
│   ├── profiling.ipynb        # FLOPs and memory profiling notebook
│   ├── evaluation.ipynb       # Accuracy evaluation scripts
├── results/
│   ├── profiling_baseline.txt # Baseline profiling results
│   ├── profiling_pruned.txt   # Pruned model profiling results
└── README.md
Getting Started
Prerequisites
Python 3.8+
PyTorch 2.0+
CUDA-enabled GPU
Installation

git clone https://github.com/ADglory/ECE1512_2024F_ProjectB_PartB_Repo_YingshunLu_Minghao-Ma
cd ECE1512_2024F_ProjectB_PartB_Repo_YingshunLu_Minghao-Ma
pip install -r requirements.txt
Usage
Run Profiling

python src/profiling.py
Evaluate Model Accuracy

python src/evaluate.py
Results and Discussion
The pruned model achieves:

31.2% FLOPs reduction and 20.2% runtime reduction.
Maintains the baseline accuracy of 52.5%.
Authors
Yingshun Lu
Minghao Ma
For questions or collaborations, please contact: your_email@example.com.

Acknowledgements
This work was part of ECE1512 at the University of Toronto, focusing on efficiency optimization in transformer models.


