    
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from math import floor
import time

def out_dimensions(conv_layer, h_in, w_in):
    '''
    This function computes the output dimension of each convolutional layers in the most general way. 
    '''
    h_out = floor((h_in + 2 * conv_layer.padding[0] - conv_layer.dilation[0] * (conv_layer.kernel_size[0] - 1) - 1) /
                  conv_layer.stride[0] + 1)
    w_out = floor((w_in + 2 * conv_layer.padding[1] - conv_layer.dilation[1] * (conv_layer.kernel_size[1] - 1) - 1) /
                  conv_layer.stride[1] + 1)
    return h_out, w_out


class CNNGodzilla(nn.Module):
    def __init__(self):
        super(CNNGodzilla, self).__init__()
        
        # First block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1, stride=1)
        h_out, w_out = out_dimensions(self.conv1, 32, 32)  # 32x32
        self.BN1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, stride=1)
        h_out, w_out = out_dimensions(self.conv2, h_out, w_out)  # 32x32
        self.BN2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        h_out, w_out = int(h_out/2), int(w_out/2)  # 16x16
        
        # Second block
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, stride=1)
        h_out, w_out = out_dimensions(self.conv3, h_out, w_out)  # 16x16
        self.BN3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, stride=1)
        h_out, w_out = out_dimensions(self.conv4, h_out, w_out)  # 16x16
        self.BN4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        h_out, w_out = int(h_out/2), int(w_out/2)  # 8x8
        
        # Third block
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, stride=1)
        h_out, w_out = out_dimensions(self.conv5, h_out, w_out)  # 8x8
        self.BN5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, stride=1)
        h_out, w_out = out_dimensions(self.conv6, h_out, w_out)  # 8x8
        self.BN6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        h_out, w_out = int(h_out/2), int(w_out/2)  # 4x4
        
        # Flatten
        # self.flatten = nn.Flatten()
        
        
        # Store final dimensions for the forward pass
        self.dimensions_final = (256, h_out, w_out)  # Should be (256, 4, 4)
        
        # Fully Connected
        self.fc1 = nn.Linear(256 * h_out * w_out, 128)  # 256 * 4 * 4 = 4096 input features
        self.BN7 = nn.BatchNorm1d(128)
        self.Dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(128, 64)
        self.BN8 = nn.BatchNorm1d(64)
        self.Dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.BN1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        x = self.BN2(x)
        x = F.gelu(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.BN3(x)
        x = F.gelu(x)
        x = self.conv4(x)
        x = self.BN4(x)
        x = F.gelu(x)
        x = self.pool2(x)

        x = self.conv5(x)
        x = self.BN5(x)
        x = F.gelu(x)
        x = self.conv6(x)
        x = self.BN6(x)
        x = F.gelu(x)
        x = self.pool3(x)

        # x = self.flatten(x) # istruzione prof
        n_channels, h, w = self.dimensions_final
        x = x.view(-1, n_channels * h * w)

        x = self.fc1(x)
        x = self.BN7(x)
        x = F.gelu(x)
        x = self.Dropout1(x)

        x = self.fc2(x)
        x = self.BN8(x)
        x = F.gelu(x)
        x = self.Dropout2(x)

        x = self.fc3(x)
        return x
    

if __name__ == '__main__':
    manual_seed = 42
    torch.manual_seed(manual_seed)    
    batch_size = 32

    transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0,0,0), std=(1,1,1))])

    dataset_train= datasets.CIFAR10(root='./data', train=True, download=True, transform=transformer)#transforms.ToTensor())
    trainloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    dataset_test= datasets.CIFAR10(root='./data', train=False, download=True, transform=transformer)
    #testloader = DataLoader(dataset_test, batch_size=len(dataset_test))
    testloader = DataLoader(dataset_test, batch_size=batch_size)    
    dataset_val, dataset_test = torch.utils.data.random_split(dataset_test, [0.5, 0.5])

    #validloader = DataLoader(dataset_val, batch_size=len(dataset_val))
    testloader = DataLoader(dataset_test, batch_size=len(dataset_test))
    validloader = DataLoader(dataset_val, batch_size=batch_size)
    '''
    Q9
    '''
    model = CNNGodzilla()
    learning_rate = 0.03
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' 
        if torch.backends.mps.is_available() else 'cpu')
    model = model.to(DEVICE)
    print("Working on", DEVICE)

    train_loss_list = []
    validation_loss_list = []
    n_epochs = 7 # with the introductions of the Dropout who avoid overfitting we can add some epochs

    for epoch in range(n_epochs):
        loss_train = 0
        for data, target in trainloader:
            # Set the model in training mode
            model.train()
            data, target = data.to(DEVICE), target.to(DEVICE)
            # Set the gradient to 0
            optimizer.zero_grad()
            # Make a prediction
            output = model(data)
            # Compute the loss function
            loss = loss_fn(output, target)
            loss_train += loss.item()
            # Backpropagation
            loss.backward()
            # Update parameters
            optimizer.step()
            
        loss_train = loss_train / len(trainloader) # Consider this alternative method of tracking training loss. 
        train_loss_list.append(loss_train)
        
        # At the end of every epoch, check the validation loss value
        with torch.no_grad():
            model.eval()
            for data, target in validloader: # Just one batch
                data, target = data.to(DEVICE), target.to(DEVICE)
                # Make a prediction
                output = model(data)
                # Compute the loss function
                validation_loss = loss_fn(output, target).item()
        print(f"Epoch {epoch + 1}: Train loss: {loss_train}, Validation loss {validation_loss}")
        validation_loss_list.append(validation_loss)

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for data, target in testloader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += target.size(0)
            n_correct += (predicted == target).sum().item()

        acc = 100.0 * n_correct / n_samples
    print("Accuracy on the test set:", acc, "%")


    plt.figure()
    plt.plot(range(n_epochs), train_loss_list)
    plt.plot(range(n_epochs), validation_loss_list)
    plt.legend(["Train loss", "Validation Loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss value")
    plt.show()
