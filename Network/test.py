from tqdm import tqdm
import torch 

from loader import load_cifar
from model import Net
from visualize import show_example_preds, visualize_weights

if __name__ == "__main__":

    batch_size = 64

    _, test_loader = load_cifar(batch_size=batch_size, num_workers=2)
    model = Net()
    model.load_state_dict(torch.load('./Network/best_model.pth'))

    model.eval()
    running_acc = 0.0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            running_acc += torch.sum(preds == labels).item()
            total += labels.size(0)
    acc = running_acc / total
    print(f"Test Accuracy: {acc:.4f}")
    show_example_preds(model, test_loader)
    visualize_weights(model)
