import torch
import torch.nn as nn  # Neural network layers

def test_model(model, test_loader, device):
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            # _, predicted = torch.max(outputs.data, 1)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        return correct / total