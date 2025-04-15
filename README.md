# Catastrophic_Forgetting

# Continual Learning on Split CIFAR-100

This project investigates various continual learning methods to mitigate catastrophic forgetting using the Split CIFAR-100 benchmark. A custom multi-head CNN model is used, with each head handling a 10-class task (total of 10 tasks). Methods include:

- ADAM optimizer with frozen heads
- Replay-based approach
- Elastic Weight Consolidation (EWC)
- Learning Without Forgetting (LwF)
- Synaptic Intelligence (SI)

---

##  File Descriptions

| File                              | Description                                                                                      |
|-----------------------------------|--------------------------------------------------------------------------------------------------|
| `adam_optimizer+freezing.py`      | Baseline training using ADAM optimizer with frozen output heads for each task.                   |
| `catastrophic_forgetting_demo.py` | Demonstration of catastrophic forgetting using naive sequential training without any mitigation. |
| `elastic_weight_consolidation.py` | Implementation of EWC for continual learning to preserve important weights.                      |
| `learning_without_forgetting.py`  | Implements LwF using distillation losses to retain knowledge of past tasks.                      |
| `replay_based_approach.py`        | Implements replay by storing and mixing a memory buffer of past samples during training.         |
| `si.py`                           | Synaptic Intelligence implementation to regularize changes in important parameters.              | 
| `vis.py`                          | Visualizes a sample image from the CIFAR-100 dataset with its label.                             |
| `F1score/`                        | Stores F1 score plots for each task and approach.                                                |
| `Graphs/`                         | Stores plots for accuracy, AUC, and confusion matrices.                                          |
| `data/`                           | Directory automatically created by torchvision for CIFAR-100 dataset.                            |

---

## Setup Instructions

1. **Install Requirements** (Python 3.7+ recommended):
    ```bash
    pip install torch torchvision scikit-learn matplotlib seaborn
    ```

2. **Run Training Scripts**

Each script runs training and evaluation on Split CIFAR-100:

```bash
python adam_optimizer+freezing.py
python replay_based_approach.py
python elastic_weight_consolidation.py
python learning_without_forgetting.py
python si.py
