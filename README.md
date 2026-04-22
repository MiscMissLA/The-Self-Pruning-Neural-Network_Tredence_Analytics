# Self-Pruning Neural Network (CIFAR-10)

[cite_start]This repository contains an implementation of a **Self-Pruning Neural Network** that dynamically learns to remove its own redundant weights during the training process[cite: 54, 55]. [cite_start]This is achieved using a differentiable gating mechanism and $L1$ regularization to identify and remove the weakest connections on the fly[cite: 55, 56].

## Project Overview
[cite_start]In standard deep learning, pruning is often a post-training step[cite: 53]. [cite_start]This project implements a **PrunableLinear** layer that associates each weight with a learnable scalar "gate"[cite: 60, 68]. [cite_start]The network learns which connections are necessary for classification on the CIFAR-10 dataset and which can be discarded to create a sparse, efficient model[cite: 58, 63, 107].

## Key Features
* [cite_start]**Custom PrunableLinear Layer**: Built from scratch using PyTorch `nn.Parameter` to handle dual-parameter optimization for both weights and gate scores[cite: 68, 72, 74].
* [cite_start]**Differentiable Gating**: Uses a Sigmoid transformation to ensure gate values remain between 0 and 1 while allowing gradient flow[cite: 77].
* [cite_start]**Sparsity-Inducing Loss**: Implements a custom loss function combining Cross-Entropy with an $L1$ penalty on gate values to encourage exact zeros[cite: 84, 87, 89, 90].
* [cite_start]**Dynamic Pruning**: The network architecture adapts during the training loop by multiplying weights by their corresponding learned gates[cite: 55, 79].

## Repository Structure
* [cite_start]`pruning_model.py`: The core implementation containing the `PrunableLinear` class, the `PruningNet` architecture, and the training/evaluation logic[cite: 110, 111, 112].
* [cite_start]`REPORT.md`: A technical analysis of the results, including the impact of the sparsity hyperparameter ($\lambda$) and gate value distributions[cite: 113].

## Methodology

### 1. Approach
[cite_start]The model uses **Differentiable Masking**[cite: 55]. Each weight $w$ is transformed into a pruned weight $w'$ via:
$$w' = w \cdot \sigma(g)$$
[cite_start]where $\sigma$ is the Sigmoid function and $g$ is the learnable gate score[cite: 77, 79].

### 2. Loss Function
The total loss is defined as:
$$Loss_{total} = Loss_{classification} + \lambda \sum |GateValues|$$
[cite_start]The $L1$ term (Sparsity Loss) provides constant pressure to drive non-essential gates toward zero[cite: 87, 89, 90, 91].

## Results
[cite_start]The trade-off between model accuracy and sparsity is controlled by the $\lambda$ hyperparameter[cite: 92, 93].

| Lambda (λ) | Test Accuracy | Sparsity Level (%) |
| :--- | :--- | :--- |
| 1e-5 | ~52% | ~12% |
| 1e-4 | ~48% | ~64% |
| 1e-3 | ~31% | ~94% |

[cite_start]*Note: Sparsity Level is defined as the percentage of weights with a gate value below 1e-2[cite: 101].*

## Edge Cases Considered
* **Layer Collapse**: Addressed by tuning $\lambda$ to prevent the penalty from over-pruning entire layers early in training, which would break gradient flow.
* **Sigmoid Saturation**: Monitored to ensure gate scores do not become so large that gradients vanish, effectively "locking" a weight in a pruned or unpruned state.
* [cite_start]**Dead Weights vs. Dead Neurons**: The current implementation prunes individual weights[cite: 60]; however, if all incoming weights to a neuron are pruned, that neuron becomes effectively inactive.
