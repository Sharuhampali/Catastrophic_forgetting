import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score
import seaborn as sns

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
        self.dataset = dataset
        self.indices = [i for i, target in enumerate(dataset.targets) if target in class_list]

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        data, label = self.dataset[actual_idx]
        return data, label

    def __len__(self):
        return len(self.indices)

# --------------------------------------------
# 2.5. Buffer dataset wrapper
# --------------------------------------------
class TupleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

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

def test(model, criterion, dataloader, device, allowed_classes):
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
            
            # Mask outputs not in allowed_classes
            mask = torch.full_like(outputs, float('-inf'))
            for cls in allowed_classes:
                mask[:, cls] = outputs[:, cls]
            outputs = mask
            
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
# 4. Main experiment: Replay-based Learning
# ----------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_epochs = 5
    batch_size = 64
    learning_rate = 0.001
    total_buffer_limit = 2000
    samples_per_class = 20

    class_order = list(range(100))
    tasks = [class_order[i:i + 10] for i in range(0, 100, 10)]

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
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

    memory_buffer = []
    class_memory = defaultdict(list)

    for task_id, task_classes in enumerate(tasks):
        print(f"\nTraining on Task {task_id+1} with classes {task_classes}")

        train_task = ClassSubset(train_dataset, task_classes)

        # Add samples to memory buffer
        for i in range(len(train_task)):
            img, lbl = train_task[i]
            if len(class_memory[lbl]) < samples_per_class:
                class_memory[lbl].append((img, lbl))

        # Limit buffer size to total_buffer_limit
        memory_buffer = [item for sublist in class_memory.values() for item in sublist]
        if len(memory_buffer) > total_buffer_limit:
            memory_buffer = random.sample(memory_buffer, total_buffer_limit)

        replay_dataset = TupleDataset(memory_buffer)
        combined_dataset = ConcatDataset([train_task, replay_dataset])
        train_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(num_epochs):
            loss = train(model, optimizer, criterion, train_loader, device)
            print(f" Task {task_id+1}, Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")

        for eval_task_id, eval_classes in enumerate(tasks[:task_id+1]):
            eval_loader = DataLoader(ClassSubset(test_dataset, eval_classes), 
                                   batch_size=batch_size, shuffle=False)
            test_loss, accuracy, y_pred, y_true, y_prob = test(model, criterion, eval_loader, 
                                                             device, allowed_classes=eval_classes)
            
            results[eval_task_id].append(accuracy)
            f1 = f1_score(y_true, y_pred, average='macro')
            
            try:
                y_true_bin = np.zeros((len(y_true), len(eval_classes)))
                for i, cls in enumerate(eval_classes):
                    y_true_bin[y_true == cls, i] = 1
                auc = roc_auc_score(y_true_bin, y_prob[:, eval_classes], 
                                   average='macro', multi_class='ovr')
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
    plt.title("Replay-Based Continual Learning on Split CIFAR-100")
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
    plt.title("F1 Score over Time with Replay")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True)
    plt.show()

    # Confusion matrix for the last task
    if len(conf_matrices) > 0:
        task_id = len(tasks) - 1
        cm = conf_matrices[task_id]
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=tasks[task_id], yticklabels=tasks[task_id])
        plt.title(f"Confusion Matrix - Task {task_id+1} (Replay)")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()