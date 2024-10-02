import torch
import torch.nn as nn  # Neural network layers
from torch.utils.tensorboard import SummaryWriter

def train_model(model, train_loader, valid_loader, criterion, optimizer, device, num_epochs):
    # writer = SummaryWriter()
    for epoch in range(num_epochs):
        model.train(True)
        running_loss = 0.

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)

            
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        running_vloss =0.
        valid_acc = 0.
        model.eval()

        with torch.no_grad():
            correct = 0
            total = 0

            for j, (vimages, vlabels) in enumerate(valid_loader):
                vimages = vimages.to(device)
                vlabels = vlabels.to(device)
                voutputs = model(vimages)
                vloss = criterion(voutputs, vlabels)
                running_vloss += vloss.item()
                _, vpredicted = voutputs.max(1)
                total += vlabels.size(0)

                correct += (vpredicted == vlabels).sum().item()



            valid_acc = correct / total
        
        print ('Epoch [{}/{}], Avg Train Loss: {:.4f}, Avg Valid Loss: {:.4f}, Valid Accuracy: {:.4f}' 
                .format(epoch+1, num_epochs, running_loss/(i+1), running_vloss/(j+1), valid_acc))
    
    # writer.flush()