import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np
from tqdm import tqdm
import copy

def get_dataset():
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    labels_map = {
        0: "T-Shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
    }

    return training_data, test_data, labels_map


def visualize_dataset(train_data, labels_map):
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(train_data), size=(1,)).item()
        img, label = train_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()


class NeuralNetwork(nn.Module):
    def __init__(self, layers=3):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = None
        if layers == 3:
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28*28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10),
            )
        elif layers == 4:
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28*28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 10),
            )
        else:
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28*28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 10),
            )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1) 
        self.pool = nn.MaxPool2d(2, 2) 
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32*7*7, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x


def evaluate(dataloader, model, criterion, device):
    total_loss, correct, total = 0.0, 0, 0
    model.eval()
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            correct += output.argmax(1).eq(target).sum().item()
            total += target.size(0)
    return total_loss / len(dataloader), 100. * correct / total


def train(models, train_loader, val_loader, criterion, device, epochs=15):
    trained_variants = {} 
    combos = []
    for m_name in models.keys():
        for opt_name in ["SGD", "Adam"]:
            combos.append((m_name, opt_name))

    results = {k: {c[0]+c[1]: [] for c in combos} for k in ["train_loss", "train_acc", "val_loss", "val_acc"]}

    for model_name, opt_type in combos:
        variant_name = f"{model_name}{opt_type}"
        print(f"\nStarting Training: {variant_name}")
        
        model = copy.deepcopy(models[model_name]).to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) if opt_type == "SGD" else optim.Adam(model.parameters(), lr=0.001)

        best_v_acc = 0.0 

        for epoch in range(epochs):
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            for data, target in pbar:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                correct += output.argmax(1).eq(target).sum().item()
                total += target.size(0)
                pbar.set_postfix(loss=loss.item())

            t_loss, t_acc = running_loss / len(train_loader), 100. * correct / total
            v_loss, v_acc = evaluate(val_loader, model, criterion, device)
            
            if v_acc > best_v_acc:
                best_v_acc = v_acc

            results["train_loss"][variant_name].append(t_loss)
            results["train_acc"][variant_name].append(t_acc)
            results["val_loss"][variant_name].append(v_loss)
            results["val_acc"][variant_name].append(v_acc)

        print(f"Finished {variant_name} | Peak Val Accuracy: {best_v_acc:.2f}%")
        trained_variants[variant_name] = model

    return results["train_loss"], results["train_acc"], results["val_loss"], results["val_acc"], trained_variants


def plot_training_results(train_losses, train_accuracies, val_losses, val_accuracies):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    metrics = [(train_losses, "Training Loss"), (val_losses, "Validation Loss"), 
               (train_accuracies, "Training Accuracy (%)"), (val_accuracies, "Validation Accuracy (%)")]
    for i, (data_dict, title) in enumerate(metrics):
        ax = axes[i//2, i%2]
        for name, values in data_dict.items():
            ax.plot(values, label=name, linewidth=2)
        ax.set_title(title, fontweight='bold'); ax.legend()
    plt.tight_layout(); plt.show()


def evaluate_and_visualize_test(models, test_loader, device, labels_map):
    model_results = []
    best_acc, best_name, best_preds, best_targets = 0, "", [], []
    
    for name, model in models.items():
        model.eval()
        cur_preds, cur_targets = [], []
        with torch.no_grad():
            for imgs, lbls in test_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(1)
                cur_preds.extend(preds.cpu().numpy()); cur_targets.extend(lbls.cpu().numpy())
        
        acc = 100 * np.mean(np.array(cur_preds) == np.array(cur_targets))
        model_results.append({"Model": name, "Accuracy": acc})
        print(f"{name:15} | Test Accuracy: {acc:.2f}%")

        if acc > best_acc:
            best_acc, best_name, best_preds, best_targets = acc, name, cur_preds, cur_targets

    plt.figure(figsize=(10, 6))
    sns.barplot(data=pd.DataFrame(model_results), x="Model", y="Accuracy", palette="viridis")
    plt.title("Test Accuracy Leaderboard", fontweight='bold'); plt.show()

    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(best_targets, best_preds), annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels_map.values(), yticklabels=labels_map.values())
    plt.title(f"Confusion Matrix: {best_name}", fontweight='bold'); plt.show()

    print(f"\nBest Model: {best_name} with {best_acc:.2f}% accuracy")

if __name__ == "__main__":    
    training_data, test_data, labels = get_dataset() 

    training_data, val_data = torch.utils.data.random_split(training_data, [50000, 10000])
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=64, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training models in {device}\n")

    base_architectures = {
        "Small" : NeuralNetwork(layers=3), 
        "Medium" : NeuralNetwork(layers=4),
        "Large" : NeuralNetwork(layers=5), 
        "CNN": SimpleCNN()
    }
    
    criterion = nn.CrossEntropyLoss()
    train_l, train_a, val_l, val_a, trained_models = train(base_architectures, train_dataloader, val_dataloader, criterion, device)
    
    plot_training_results(train_l, train_a, val_l, val_a)
    evaluate_and_visualize_test(trained_models, test_dataloader, device, labels)