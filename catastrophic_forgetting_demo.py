# #Step1
# #Catastrophic forgetting without any mitigation techniques -> rewrites the weight everytime a new task is trained on
# #incremental class learning:
# # ->Each task is a subset of 10 new classes.
# # ->But we are using the same model that maps to all 100 class labels from the very start.
# # ->The model is trained on only 10 classes at a time, yet asked to classify across 100 labels.
# # ->During testing, accuracy is measured on each task's class subset, using the full model.

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score
import seaborn as sns
from collections import defaultdict

# ----------------------------------------
# 1. Define a Simple CNN for CIFAR-100
# ----------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=100):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --------------------------------------------
# 2. Custom subset that keeps original labels
# --------------------------------------------
class ClassSubset(Dataset):
    def __init__(self, dataset, class_list):
        self.indices = [i for i, target in enumerate(dataset.targets) if target in class_list]
        self.subset = Subset(dataset, self.indices)

    def __getitem__(self, idx):
        return self.subset[idx]

    def __len__(self):
        return len(self.subset)

# ----------------------------------------
# 3. Training and testing functions
# ----------------------------------------
def train(model, optimizer, criterion, dataloader, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(dataloader.dataset)

def test(model, criterion, dataloader, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())

    return (test_loss / len(dataloader.dataset),
            correct / len(dataloader.dataset),
            np.array(all_preds),
            np.array(all_labels),
            np.array(all_probs))

# ----------------------------------------
# 4. Main experiment: Split CIFAR-100
# ----------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_epochs = 5
    batch_size = 64
    learning_rate = 0.001

    class_order = list(range(100))
    tasks = [class_order[i:i + 10] for i in range(0, 100, 10)]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    train_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=False, transform=transform, download=True)

    results = {i: [] for i in range(len(tasks))}
    f1_results = {i: [] for i in range(len(tasks))}
    auc_results = {i: [] for i in range(len(tasks))}
    conf_matrices = {}

    model = SimpleCNN(num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for task_id, task_classes in enumerate(tasks):
        print(f"\nTraining on Task {task_id+1} with classes {task_classes}")

        train_task = ClassSubset(train_dataset, task_classes)
        test_task = ClassSubset(test_dataset, task_classes)
        train_loader = DataLoader(train_task, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_task, batch_size=batch_size, shuffle=False)

        for epoch in range(num_epochs):
            loss = train(model, optimizer, criterion, train_loader, device)
            print(f" Task {task_id+1}, Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")

        for eval_task_id, eval_classes in enumerate(tasks[:task_id+1]):
            eval_loader = DataLoader(ClassSubset(test_dataset, eval_classes), batch_size=batch_size, shuffle=False)
            test_loss, accuracy, y_pred, y_true, y_prob = test(model, criterion, eval_loader, device)

            results[eval_task_id].append(accuracy)
            f1 = f1_score(y_true, y_pred, average='macro')

            try:
                y_true_bin = np.zeros((len(y_true), len(eval_classes)))
                y_true_bin[np.arange(len(y_true)), y_true - min(eval_classes)] = 1
                auc = roc_auc_score(y_true_bin, y_prob[:, eval_classes], average='macro', multi_class='ovr')
            except ValueError:
                auc = float('nan')

            f1_results[eval_task_id].append(f1)
            auc_results[eval_task_id].append(auc)

            if task_id == len(tasks) - 1 and eval_task_id == len(tasks) - 1:
                cm = confusion_matrix(y_true, y_pred, labels=eval_classes)
                conf_matrices[eval_task_id] = cm

            print(f"  -> Eval on Task {eval_task_id+1} (classes {eval_classes}): "
                  f"Accuracy = {accuracy*100:.2f}%, F1 = {f1:.2f}, AUC = {auc:.2f}")

    # Final combined accuracy plot
    plt.figure(figsize=(8, 6))
    for task_id in range(len(tasks)):
        acc = results[task_id]
        acc_padded = acc + [np.nan] * (len(tasks) - len(acc))
        plt.plot(range(1, len(tasks) + 1), acc_padded, marker='o',
                 label=f"Task {task_id+1} (classes {tasks[task_id]})")
    plt.xlabel("Task Sequence (Training Order)")
    plt.ylabel("Test Accuracy")
    plt.title("Catastrophic Forgetting in Split CIFAR-100")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True)
    plt.show()

    # Final F1 score plot
    plt.figure(figsize=(8, 6))
    for task_id in range(len(tasks)):
        f1 = f1_results[task_id]
        f1_padded = f1 + [np.nan] * (len(tasks) - len(f1))
        plt.plot(range(1, len(tasks) + 1), f1_padded, marker='o',
                 label=f"Task {task_id+1} (classes {tasks[task_id]})")
    plt.xlabel("Task Sequence (Training Order)")
    plt.ylabel("F1 Score")
    plt.title("F1 Score over Time")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True)
    plt.show()

    # Confusion matrix 
    if len(conf_matrices) > 0:
        task_id = len(tasks) - 1
        cm = conf_matrices[task_id]
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=tasks[task_id], yticklabels=tasks[task_id])
        plt.title(f"Confusion Matrix - Task {task_id+1}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
