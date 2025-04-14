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
import copy

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
# 2.5. Buffer dataset wrapper for DER
# --------------------------------------------
class TupleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]  # (img, label, logits)

    def __len__(self):
        return len(self.data)

# --------------------------------------------
# 2.6. Wrap original training dataset to match replay format
# --------------------------------------------
class TrainWrapper(Dataset):
    def __init__(self, dataset, prev_model, device):
        self.dataset = dataset
        self.prev_model = prev_model
        self.device = device

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        with torch.no_grad():
            self.prev_model.eval()
            input_img = img.unsqueeze(0).to(self.device)
            logits = self.prev_model(input_img).squeeze(0).cpu()
        return img, label, logits

    def __len__(self):
        return len(self.dataset)

# ----------------------------------------
# 3. Training and testing functions
# ----------------------------------------
def train(model, optimizer, ce_criterion, kl_criterion, dataloader, device):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        inputs = batch[0].to(device)
        labels = batch[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = ce_criterion(outputs, labels)

        if len(batch) == 3:
            old_logits = batch[2].to(device)
            distill_loss = kl_criterion(torch.log_softmax(outputs, dim=1), torch.softmax(old_logits, dim=1))
            loss += distill_loss

        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(dataloader.dataset)

def test(model, criterion, dataloader, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
    return test_loss / len(dataloader.dataset), correct / len(dataloader.dataset)

# ----------------------------------------
# 4. Main experiment: DER-based Replay
# ----------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_epochs = 5
    batch_size = 64
    learning_rate = 0.001
    replay_buffer_size = 2000
    samples_per_class = 20

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

    model = SimpleCNN(num_classes=100).to(device)
    ce_criterion = nn.CrossEntropyLoss()
    kl_criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    memory_buffer = []
    class_memory = defaultdict(list)
    prev_model = copy.deepcopy(model)

    for task_id, task_classes in enumerate(tasks):
        print(f"\nTraining on Task {task_id+1} with classes {task_classes}")

        train_task = ClassSubset(train_dataset, task_classes)

        for i in range(len(train_task)):
            img, lbl = train_task[i]
            if len(class_memory[lbl]) < samples_per_class:
                with torch.no_grad():
                    prev_model.eval()
                    img_input = img.unsqueeze(0).to(device)
                    logits = prev_model(img_input).squeeze(0).cpu()
                    class_memory[lbl].append((img, lbl, logits))

        memory_buffer = [item for sublist in class_memory.values() for item in sublist]
        replay_dataset = TupleDataset(memory_buffer)
        train_wrapped = TrainWrapper(train_task, prev_model, device)
        combined_dataset = ConcatDataset([train_wrapped, replay_dataset])
        train_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(num_epochs):
            loss = train(model, optimizer, ce_criterion, kl_criterion, train_loader, device)
            print(f" Task {task_id+1}, Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")

        prev_model = copy.deepcopy(model)

        for eval_task_id, eval_classes in enumerate(tasks[:task_id+1]):
            eval_loader = DataLoader(ClassSubset(test_dataset, eval_classes), batch_size=batch_size, shuffle=False)
            test_loss, accuracy = test(model, ce_criterion, eval_loader, device)
            results[eval_task_id].append(accuracy)
            print(f"  -> Eval on Task {eval_task_id+1} (classes {eval_classes}): "
                  f"Accuracy = {accuracy*100:.2f}%")

    for task_id in range(len(tasks)):
        plt.figure(figsize=(6, 4))
        acc = results[task_id]
        acc_padded = acc + [np.nan] * (len(tasks) - len(acc))
        x_values = list(range(1, len(tasks) + 1))

        plt.plot(x_values, acc_padded, marker='o',
                 label=f"Task {task_id+1} (classes {tasks[task_id]})")
        plt.xlabel("Task Sequence (Training Order)")
        plt.ylabel("Test Accuracy")
        plt.title(f"Accuracy for Task {task_id+1} over Time")
        plt.ylim(0, 1.05)
        plt.legend()
        plt.grid(True)
        plt.show()

    plt.figure(figsize=(8, 6))
    for task_id in range(len(tasks)):
        acc = results[task_id]
        acc_padded = acc + [np.nan] * (len(tasks) - len(acc))
        plt.plot(range(1, len(tasks) + 1), acc_padded, marker='o',
                 label=f"Task {task_id+1} (classes {tasks[task_id]})")
    plt.xlabel("Task Sequence (Training Order)")
    plt.ylabel("Test Accuracy")
    plt.title("Catastrophic Forgetting with DER Replay")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()

