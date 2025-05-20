import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader

from loader import load_cifar
from model import Net
from visualize import plot_loss_acc

if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    lr = 1e-3
    weight_decay = 1e-4
    batch_size = 64

    num_epochs = 40

    train_loader, _ = load_cifar(batch_size=64, num_workers=2)

    train_size = int(len(train_loader.dataset) * 0.8)
    val_size = len(train_loader.dataset) - train_size
    train_dataset, val_dataset = random_split(train_loader.dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    start_time = time.time()

    model = Net()
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer = optim.RMSprop(model.parameters(), lr=lr, alpha=0.99, weight_decay=weight_decay)

    best_val_acc = 0.0
    patience = 5
    no_improve_epochs = 0
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        total = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_acc += torch.sum(preds == labels).item()
            total += labels.size(0)

        loss = running_loss / len(train_loader)
        acc = running_acc / total
        train_losses.append(loss)
        train_accuracies.append(acc)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss:.4f}, Train Acc: {acc:.4f}")

        model.eval()
        running_loss = 0.0
        running_acc = 0.0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                running_acc += torch.sum(preds == labels).item()
                total += labels.size(0)
            
        loss = running_loss / len(val_loader)
        acc = running_acc / total
        val_losses.append(loss)
        val_accuracies.append(acc)
        print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {loss:.4f}, Val Acc: {acc:.4f}")

        if acc > best_val_acc:
            best_val_acc = acc
            no_improve_epochs = 0
            torch.save(model.state_dict(), './Network/best_model.pth')
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f'Early stopping at epoch {epoch+1} with best val acc: {best_val_acc:.4f}')
                break
        # scheduler.step()
        print('-' * 30)

    end_time = time.time()
    print(f'Training completed in {(end_time - start_time)/60:.2f} minutes')

    plot_loss_acc(train_losses, train_accuracies, val_losses, val_accuracies)

