### **Self-Pruning Neural Network (CIFAR-10)**

This repository contains an implementation of a **Self-Pruning Neural Network** that dynamically learns to remove its own redundant weights during the training process. This is achieved using a differentiable gating mechanism and L1 regularization to identify and remove the weakest connections on the fly.

---

### **Project Overview**
In standard deep learning, pruning is often a post-training step. This project implements a **PrunableLinear** layer that associates each weight with a learnable scalar "gate." The network learns which connections are necessary for classification on the CIFAR-10 dataset and which can be discarded to create a sparse, efficient model.

---

### **Key Features**
* **Custom PrunableLinear Layer**: Built from scratch using PyTorch `nn.Parameter` to handle dual-parameter optimization for both weights and gate scores.
* **Differentiable Gating**: Uses a Sigmoid transformation to ensure gate values remain between 0 and 1 while allowing gradient flow.
* **Sparsity-Inducing Loss**: Implements a custom loss function combining Cross-Entropy with an L1 penalty on gate values to encourage exact zeros.
* **Dynamic Pruning**: The network architecture adapts during the training loop by multiplying weights by their corresponding learned gates.

---

### **Repository Structure**
* `pruning_model.py`: The core implementation containing the `PrunableLinear` class, the `PruningNet` architecture, and the training/evaluation logic.
* `REPORT.md`: A technical analysis of the results, including the impact of the sparsity hyperparameter (λ) and gate value distributions.

---

### **Methodology**

#### **1. Approach**
The model uses **Differentiable Masking**. Each weight $w$ is transformed into a pruned weight $w'$ via:
$$w' = w \cdot \sigma(g)$$
where $\sigma$ is the Sigmoid function and $g$ is the learnable gate score.

#### **2. Loss Function**
The total loss is defined as:
$$Loss_{total} = Loss_{classification} + \lambda \sum |GateValues|$$
The L1 term (Sparsity Loss) provides constant pressure to drive non-essential gates toward zero.

---

### **Results**
The trade-off between model accuracy and sparsity is controlled by the λ hyperparameter.

| Lambda (λ) | Test Accuracy | Sparsity Level (%) |
| :--- | :--- | :--- |
| 1e-5 | ~52% | ~12% |
| 1e-4 | ~48% | ~64% |
| 1e-3 | ~31% | ~94% |

*Note: Sparsity Level is defined as the percentage of weights with a gate value below 1e-2.*

---

### **Edge Cases Considered**
* **Layer Collapse**: Addressed by tuning λ to prevent the penalty from over-pruning entire layers early in training, which would break gradient flow.
* **Sigmoid Saturation**: Monitored to ensure gate scores do not become so large that gradients vanish, effectively "locking" a weight in a pruned or unpruned state.
* **Dead Weights vs. Dead Neurons**: The current implementation prunes individual weights; however, if all incoming weights to a neuron are pruned, that neuron becomes effectively inactive.

---

### **Getting Started**

**Installation**
```bash
pip install torch torchvision matplotlib numpy
```

**Running the Analysis**
```bash
python pruning_model.py
```
