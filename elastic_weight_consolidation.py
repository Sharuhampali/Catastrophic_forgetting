
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader, Subset, Dataset
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score

# # ----------------------------------------
# # 1. Define a Base CNN (Feature Extractor)
# # ----------------------------------------
# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.fc1 = nn.Linear(64 * 8 * 8, 256)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x = self.pool(self.relu(self.conv1(x)))
#         x = self.pool(self.relu(self.conv2(x)))
#         x = x.view(x.size(0), -1)
#         x = self.relu(self.fc1(x))
#         return x

# # ----------------------------------------
# # 2. Multi-Head Classifier for Tasks
# # ----------------------------------------
# class MultiHeadCNN(nn.Module):
#     def __init__(self, base_model, task_count, hidden_dim=256):
#         super(MultiHeadCNN, self).__init__()
#         self.feature_extractor = base_model
#         self.heads = nn.ModuleList([nn.Linear(hidden_dim, 10) for _ in range(task_count)])

#     def forward(self, x, task_id):
#         x = self.feature_extractor(x)
#         return self.heads[task_id](x)

# # --------------------------------------------
# # 3. Custom subset that remaps class indices
# # --------------------------------------------
# class ClassSubset(Dataset):
#     def __init__(self, dataset, class_list):
#         self.indices = [i for i, target in enumerate(dataset.targets) if target in class_list]
#         self.subset = Subset(dataset, self.indices)
#         self.class_map = {cls: idx for idx, cls in enumerate(class_list)}

#     def __getitem__(self, idx):
#         data, target = self.subset[idx]
#         return data, self.class_map[int(target)]

#     def __len__(self):
#         return len(self.subset)

# # --------------------------------------------
# # 4. EWC Regularization
# # --------------------------------------------
# class EWC:
#     def __init__(self, model, dataloader, device):
#         self.model = model
#         self.device = device
#         self.dataloader = dataloader
#         self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
#         self._means = {}
#         self._precision_matrices = self._diag_fisher()
#         for n, p in self.params.items():
#             self._means[n] = p.clone().detach()

#     def _diag_fisher(self):
#         precision_matrices = {}
#         for n, p in self.params.items():
#             precision_matrices[n] = torch.zeros_like(p)

#         self.model.eval()
#         for input, target in self.dataloader:
#             input, target = input.to(self.device), target.to(self.device)
#             self.model.zero_grad()
#             output = self.model(input, task_id=0)  # task_id must be valid for current head
#             loss = nn.functional.cross_entropy(output, target)
#             loss.backward()
#             for n, p in self.model.named_parameters():
#                 if p.grad is not None:
#                     precision_matrices[n].data += p.grad.data ** 2 / len(self.dataloader)
#         return precision_matrices

#     def penalty(self, model):
#         loss = 0
#         for n, p in model.named_parameters():
#             if n in self._precision_matrices:
#                 _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
#                 loss += _loss.sum()
#         return loss

# # ----------------------------------------
# # 5. Training and testing functions
# # ----------------------------------------
# def train(model, optimizer, criterion, dataloader, device, task_id, ewc=None, lambda_ewc=0.4):
#     model.train()
#     running_loss = 0.0
#     for inputs, labels in dataloader:
#         inputs, labels = inputs.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(inputs, task_id)
#         loss = criterion(outputs, labels)
#         if ewc is not None:
#             loss += lambda_ewc * ewc.penalty(model)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item() * inputs.size(0)
#     return running_loss / len(dataloader.dataset)

# def test(model, criterion, dataloader, device, task_id):
#     model.eval()
#     test_loss = 0.0
#     correct = 0
#     with torch.no_grad():
#         for inputs, labels in dataloader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs, task_id)
#             loss = criterion(outputs, labels)
#             test_loss += loss.item() * inputs.size(0)
#             _, predicted = torch.max(outputs, 1)
#             correct += (predicted == labels).sum().item()
#     return test_loss / len(dataloader.dataset), correct / len(dataloader.dataset)

# # ----------------------------------------
# # 6. Main experiment with EWC
# # ----------------------------------------
# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     num_epochs = 10
#     batch_size = 64
#     learning_rate = 0.001
#     lambda_ewc = 100

#     class_order = list(range(100))
#     tasks = [class_order[i:i + 10] for i in range(0, 100, 10)]

#     transform = transforms.Compose([
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomCrop(32, padding=4),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
#     ])

#     train_dataset = torchvision.datasets.CIFAR100(
#         root='./data', train=True, transform=transform, download=True)
#     test_dataset = torchvision.datasets.CIFAR100(
#         root='./data', train=False, transform=transform, download=True)

#     results = {i: [] for i in range(len(tasks))}
#     f1_results = {i: [] for i in range(len(tasks))}
#     auc_results = {i: [] for i in range(len(tasks))}
#     conf_matrices = {}

#     base_feature_model = SimpleCNN()
#     model = MultiHeadCNN(base_feature_model, task_count=len(tasks)).to(device)

#     criterion = nn.CrossEntropyLoss()
#     ewc_list = []

#     for task_id, task_classes in enumerate(tasks):
#         print(f"\nTraining on Task {task_id+1} with classes {task_classes}")

#         train_task = ClassSubset(train_dataset, task_classes)
#         test_task = ClassSubset(test_dataset, task_classes)
#         train_loader = DataLoader(train_task, batch_size=batch_size, shuffle=True)
#         test_loader = DataLoader(test_task, batch_size=batch_size, shuffle=False)

#         optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#         for epoch in range(num_epochs):
#             ewc_penalty = ewc_list[-1] if ewc_list else None
#             loss = train(model, optimizer, criterion, train_loader, device, task_id, ewc=ewc_penalty, lambda_ewc=lambda_ewc)
#             print(f" Task {task_id+1}, Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")

#         ewc = EWC(model, train_loader, device)
#         ewc_list.append(ewc)

#         for eval_task_id, eval_classes in enumerate(tasks[:task_id+1]):
#             eval_loader = DataLoader(ClassSubset(test_dataset, eval_classes), batch_size=batch_size, shuffle=False)
#             test_loss, accuracy = test(model, criterion, eval_loader, device, eval_task_id)
#             results[eval_task_id].append(accuracy)
#             print(f"  -> Eval on Task {eval_task_id+1} (classes {eval_classes}): Accuracy = {accuracy*100:.2f}%")

#     plt.figure(figsize=(8, 6))
#     for task_id in range(len(tasks)):
#         acc = results[task_id]
#         acc_padded = acc + [np.nan] * (len(tasks) - len(acc))
#         plt.plot(range(1, len(tasks) + 1), acc_padded, marker='o',
#                  label=f"Task {task_id+1} (classes {tasks[task_id]})")
#     plt.xlabel("Task Sequence (Training Order)")
#     plt.ylabel("Test Accuracy")
#     plt.title("EWC Mitigation of Catastrophic Forgetting")
#     plt.ylim(0, 1.05)
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# if __name__ == "__main__":
#     main()

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

# ----------------------------------------
# 1. Define a Base CNN (Feature Extractor)
# ----------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return x

# ----------------------------------------
# 2. Multi-Head Classifier for Tasks
# ----------------------------------------
class MultiHeadCNN(nn.Module):
    def __init__(self, base_model, task_count, hidden_dim=256):
        super(MultiHeadCNN, self).__init__()
        self.feature_extractor = base_model
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 10) for _ in range(task_count)])

    def forward(self, x, task_id):
        x = self.feature_extractor(x)
        return self.heads[task_id](x)

# --------------------------------------------
# 3. Custom subset that remaps class indices
# --------------------------------------------
class ClassSubset(Dataset):
    def __init__(self, dataset, class_list):
        self.indices = [i for i, target in enumerate(dataset.targets) if target in class_list]
        self.subset = Subset(dataset, self.indices)
        self.class_map = {cls: idx for idx, cls in enumerate(class_list)}

    def __getitem__(self, idx):
        data, target = self.subset[idx]
        return data, self.class_map[int(target)]

    def __len__(self):
        return len(self.subset)

# --------------------------------------------
# 4. EWC Regularization
# --------------------------------------------
class EWC:
    def __init__(self, model, dataloader, device):
        self.model = model
        self.device = device
        self.dataloader = dataloader
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()
        for n, p in self.params.items():
            self._means[n] = p.clone().detach()

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in self.params.items():
            precision_matrices[n] = torch.zeros_like(p)

        self.model.eval()
        for input, target in self.dataloader:
            input, target = input.to(self.device), target.to(self.device)
            self.model.zero_grad()
            output = self.model(input, task_id=0)  # task_id must be valid for current head
            loss = nn.functional.cross_entropy(output, target)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    precision_matrices[n].data += p.grad.data ** 2 / len(self.dataloader)
        return precision_matrices

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if n in self._precision_matrices:
                _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
                loss += _loss.sum()
        return loss

# ----------------------------------------
# 5. Training and testing functions
# ----------------------------------------
def train(model, optimizer, criterion, dataloader, device, task_id, ewc=None, lambda_ewc=0.4):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs, task_id)
        loss = criterion(outputs, labels)
        if ewc is not None:
            loss += lambda_ewc * ewc.penalty(model)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(dataloader.dataset)

def test(model, criterion, dataloader, device, task_id):
    model.eval()
    test_loss = 0.0
    correct = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs, task_id)
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
# 6. Main experiment with EWC
# ----------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_epochs = 10
    batch_size = 64
    learning_rate = 0.001
    lambda_ewc = 100  # Higher lambda means stronger EWC penalty

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

    base_feature_model = SimpleCNN()
    model = MultiHeadCNN(base_feature_model, task_count=len(tasks)).to(device)
    criterion = nn.CrossEntropyLoss()
    ewc_list = []

    for task_id, task_classes in enumerate(tasks):
        print(f"\nTraining on Task {task_id+1} with classes {task_classes}")

        train_task = ClassSubset(train_dataset, task_classes)
        test_task = ClassSubset(test_dataset, task_classes)
        train_loader = DataLoader(train_task, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_task, batch_size=batch_size, shuffle=False)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            ewc_penalty = ewc_list[-1] if ewc_list else None
            loss = train(model, optimizer, criterion, train_loader, device, task_id, ewc=ewc_penalty, lambda_ewc=lambda_ewc)
            print(f" Task {task_id+1}, Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")

        ewc = EWC(model, train_loader, device)
        ewc_list.append(ewc)

        for eval_task_id, eval_classes in enumerate(tasks[:task_id+1]):
            eval_loader = DataLoader(ClassSubset(test_dataset, eval_classes), batch_size=batch_size, shuffle=False)
            test_loss, accuracy, y_pred, y_true, y_prob = test(model, criterion, eval_loader, device, eval_task_id)

            results[eval_task_id].append(accuracy)
            f1 = f1_score(y_true, y_pred, average='macro')

            try:
                y_true_bin = np.zeros((len(y_true), 10))  # 10 classes per task
                y_true_bin[np.arange(len(y_true)), y_true] = 1
                auc = roc_auc_score(y_true_bin, y_prob, average='macro', multi_class='ovr')
            except ValueError:
                auc = float('nan')

            f1_results[eval_task_id].append(f1)
            auc_results[eval_task_id].append(auc)

            if task_id == len(tasks) - 1 and eval_task_id == len(tasks) - 1:
                cm = confusion_matrix(y_true, y_pred)
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
    plt.title("EWC Mitigation of Catastrophic Forgetting")
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
    plt.title("F1 Score over Time with EWC")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True)
    plt.show()

    # Confusion matrix for the last task
    if len(conf_matrices) > 0:
        task_id = len(tasks) - 1
        cm = conf_matrices[task_id]
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix - Task {task_id+1} (EWC)")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()