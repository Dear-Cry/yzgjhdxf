import numpy as np
import torch
import matplotlib.pyplot as plt

def plot_loss_acc(train_losses, train_accuracies, val_losses, val_accuracies):
    epochs = np.arange(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig('./Network/loss_acc.png')

def show_example_preds(model, loader):
    classes = {0: 'plane', 1: 'car', 2: 'bird', 3: 'cat', 4: 'deer',
               5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
    images, labels = next(iter(loader))
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

    plt.figure(figsize=(12, 12))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        image = images[i].numpy().transpose((1, 2, 0))
        image = image * 0.5 + 0.5
        plt.imshow(image)
        plt.title(f'Pred: {classes[preds[i].item()]}, True: {classes[labels[i].item()]}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('./Network/example_preds.png')

def visualize_weights(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            weights = module.weight.data.cpu().numpy()
            num_filters = weights.shape[0]

            plt.figure(figsize=(12, 12))
            for i in range(num_filters):
                plt.subplot(6, 6, i + 1)
                weight = weights[i].transpose((1, 2, 0))
                weight = (weight - weight.min()) / (weight.max() - weight.min())
                plt.imshow(weight)
                plt.axis('off')
            plt.suptitle(f'Weights of {name}')
            plt.tight_layout()
            plt.savefig(f'./Network/weights_{name}.png')
            break