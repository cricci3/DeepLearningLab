'''
Assignment 2
Student: CLAUDIO RICCI
'''

# *** Packages ***
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


'''
Q2
'''
def get_labels_from_loader(loader):
    '''
    Function to extract all the labels associated to an image in the DataLoader
    The labels are extracted as Numpy arrays
    '''
    all_labels = []
    for _, labels in loader:
        all_labels.extend(labels.numpy())
    return np.array(all_labels)


'''
Q6
'''
def out_dimensions(conv_layer, h_in, w_in):
    '''
    This function computes the output dimension of each convolutional layers in the most general way. 
    '''
    h_out = floor((h_in + 2 * conv_layer.padding[0] - conv_layer.dilation[0] * (conv_layer.kernel_size[0] - 1) - 1) /
                  conv_layer.stride[0] + 1)
    w_out = floor((w_in + 2 * conv_layer.padding[1] - conv_layer.dilation[1] * (conv_layer.kernel_size[1] - 1) - 1) /
                  conv_layer.stride[1] + 1)
    return h_out, w_out


class CNNBasic(nn.Module):
    '''
    Simple CNN model definition
    '''
    def __init__(self):
        super(CNNBasic, self).__init__()
        # First Convolutional Block (Conv-Conv-Activ-Pool)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=0, stride=1)
        h_out, w_out = out_dimensions(self.conv1, 32, 32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=0, stride=1)
        h_out, w_out = out_dimensions(self.conv2, h_out, w_out)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        h_out, w_out = int(h_out/2), int(w_out/2)
        
        # Second Convolutional Block (Conv-Conv-Activ-Pool)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=0, stride=1)
        h_out, w_out = out_dimensions(self.conv3, h_out, w_out)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), padding=0, stride=1)
        h_out, w_out = out_dimensions(self.conv4, h_out, w_out)
        h_out, w_out = int(h_out/2), int(w_out/2)
        
        # Store final dimensions for the forward pass
        self.dimensions_final = (64, h_out, w_out)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * h_out * w_out, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)


    def forward(self, x):
        # First Convolutional Block (Conv-Conv-Activ-Pool)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Second Convolutional Block (Conv-Conv-Activ-Pool)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        n_channels, h, w = self.dimensions_final
        # Flatten
        x = x.view(-1, n_channels * h * w)
        
        # 3 fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


'''
Q9
'''
class CNNGodzilla(nn.Module):
    '''
    Advanced CNN model definition
    '''
    def __init__(self):
        super(CNNGodzilla, self).__init__()
        
        # First Convolutional Block (Conv-BN-Activ-Conv-BN-Activ-Pool)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1, stride=1)
        h_out, w_out = out_dimensions(self.conv1, 32, 32)
        self.BN1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, stride=1)
        h_out, w_out = out_dimensions(self.conv2, h_out, w_out)
        self.BN2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        h_out, w_out = int(h_out/2), int(w_out/2)
        
        # Second Convolutional Block (Conv-BN-Activ-Conv-BN-Activ-Pool)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, stride=1)
        h_out, w_out = out_dimensions(self.conv3, h_out, w_out)
        self.BN3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, stride=1)
        h_out, w_out = out_dimensions(self.conv4, h_out, w_out) 
        self.BN4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        h_out, w_out = int(h_out/2), int(w_out/2)
        
        # Third Convolutional Block (Conv-BN-Activ-Conv-BN-Activ-Pool)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, stride=1)
        h_out, w_out = out_dimensions(self.conv5, h_out, w_out)
        self.BN5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, stride=1)
        h_out, w_out = out_dimensions(self.conv6, h_out, w_out)
        self.BN6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        h_out, w_out = int(h_out/2), int(w_out/2)
        
        # Store final dimensions for the forward pass
        self.dimensions_final = (256, h_out, w_out)
        
        # First Fully Connected Block (FC-BN-Activ-Dropout)
        self.fc1 = nn.Linear(256 * h_out * w_out, 128)
        self.BN7 = nn.BatchNorm1d(128)
        self.Dropout1 = nn.Dropout(0.5)

        # Second Fully Connected Block (FC-BN-Activ-Dropout)
        self.fc2 = nn.Linear(128, 64)
        self.BN8 = nn.BatchNorm1d(64)
        self.Dropout2 = nn.Dropout(0.5)

        # Third Fully Connected Layer
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # First Convolutional Block (Conv-BN-Activ-Conv-BN-Activ-Pool)
        x = self.conv1(x)
        x = self.BN1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        x = self.BN2(x)
        x = F.gelu(x)
        x = self.pool1(x)

        # Second Convolutional Block (Conv-BN-Activ-Conv-BN-Activ-Pool)
        x = self.conv3(x)
        x = self.BN3(x)
        x = F.gelu(x)
        x = self.conv4(x)
        x = self.BN4(x)
        x = F.gelu(x)
        x = self.pool2(x)

        # Third Convolutional Block (Conv-BN-Activ-Conv-BN-Activ-Pool)
        x = self.conv5(x)
        x = self.BN5(x)
        x = F.gelu(x)
        x = self.conv6(x)
        x = self.BN6(x)
        x = F.gelu(x)
        x = self.pool3(x)

        n_channels, h, w = self.dimensions_final
        
        # Flatten 
        x = x.view(-1, n_channels * h * w)

        # First Fully Connected Block (FC-BN-Activ-Dropout)
        x = self.fc1(x)
        x = self.BN7(x)
        x = F.gelu(x)
        x = self.Dropout1(x)

        # Second Fully Connected Block (FC-BN-Activ-Dropout)
        x = self.fc2(x)
        x = self.BN8(x)
        x = F.gelu(x)
        x = self.Dropout2(x)

        # Third Fully Connected Layer
        x = self.fc3(x)
        return x



if __name__ == "__main__":
    print("Hello World!")

    '''
    DON'T MODIFY THE SEED!
    '''
    # Set the seed for reproducibility
    manual_seed = 42
    torch.manual_seed(manual_seed)

    
    '''
    Q2 - Code
    '''
    batch_size = 32

    # Defined transformations for the dataset: convert images to tensors and normalize them
    transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0,0,0), std=(1,1,1))])

    # Load CIFAR-10 training and test dataset with defined transformations
    dataset_train= datasets.CIFAR10(root='./data', train=True, download=True, transform=transformer)#transforms.ToTensor())
    trainloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    dataset_test= datasets.CIFAR10(root='./data', train=False, download=True, transform=transformer)
    testloader = DataLoader(dataset_test, batch_size=batch_size)

    # Dictionary to map class indices to class names
    classes_map = {
        0 : 'plane',
        1 : 'car',
        2 : 'bird',
        3 : 'cat',
        4 : 'deer',
        5 : 'dog',
        6 : 'frog',
        7 : 'horse',
        8 : 'ship',
        9 : 'truck'
    }

    # Figure to display one example image per class
    figure = plt.figure(figsize=(8, 4))
    cols, rows = 5, 2

    # Dictionary to keep track of one image for each class
    class_examples = {}

    # Loop through training dataset to find one image per class
    for i, (img, label) in enumerate(dataset_train):
            # dataset return img, label 
            # enumerate return i
        if label not in class_examples and len(class_examples) < 10:
            class_examples[label] = img
        if len(class_examples) == 10:
            break

    # Plot one image per classe in the already defined figure
    for i, (label, img) in enumerate(class_examples.items(), 1):
        figure.add_subplot(rows, cols, i)
        img = img.permute(1, 2, 0)
        plt.title(classes_map[label])
        plt.axis("off")
        plt.imshow(img)
    plt.show()

    # Get labels from training and test datasets, storing them as arrays
    train_label = get_labels_from_loader(trainloader)
    test_label = get_labels_from_loader(testloader)

    # Calculate the distribution of classes in both training and test datasets
    _, train_counts = np.unique(train_label, return_counts=True)
    _, test_counts = np.unique(test_label, return_counts=True)

    # Comparative bar chart showing class distributions
    bar_width = 0.35
    index = np.arange(len(class_examples))

    plt.bar(index, train_counts, bar_width, label='Train', color='deepskyblue')
    plt.bar(index + bar_width, test_counts, bar_width, label='Test', color='lightgreen')
    plt.xlabel('Classes')
    plt.ylabel('Number of Images')
    plt.title('Class Distribution in Train and Test Sets')
    plt.xticks(index + bar_width / 2, class_examples)
    plt.legend()
    plt.grid()
    plt.show()

    # Also print the values
    print("Numero di immagini per classe:")
    print(f"{'Classe':<10} {'Training':<10} {'Test':<10}")
    print("-" * 30)
    for i in range(10):
        print(f"{classes_map[i]:<10} {train_counts[i]:<10} {test_counts[i]:<10}")


    '''
    Q3
    '''
    # Get a batch of images from the training loader and extract the first image
    trainiter = iter(trainloader)
    train_images, _ = next(trainiter)
    first_image = train_images[0]

    # Display the type of the dataset element
    print(f"Type of each element of the dataset: {first_image.type}")

    # Display the shape of the image tensor (channels, height, width)
    print(f"Shape of the image tensor: {first_image.shape}")  # (C, H, W)

    # Separate the shape into channels, height, and width
    channels, height, width = first_image.shape
    print(f"Width: {width}, Height: {height}, Channels: {channels}")
    

    '''
    Q5
    '''
    # Split the test dataset into validation and test set (50-50%) and define DataLoaders
    dataset_val, dataset_test = torch.utils.data.random_split(dataset_test, [0.5, 0.5])

    testloader = DataLoader(dataset_test, batch_size=len(dataset_test))
    validloader = DataLoader(dataset_val, batch_size=batch_size)

    print(len(trainloader), len(validloader))


    '''
    Q7
    '''
    # Initialize the first CNN model and set hyperparameters
    model = CNNBasic()
    learning_rate = 0.03
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' 
        if torch.backends.mps.is_available() else 'cpu')
    model = model.to(DEVICE)
    print("Working on", DEVICE)


    # Number of epochs and size of steps
    n_epochs = 4
    n_steps = 50

    # Lists to store accuracy and loss metrics every n_steps
    train_acc_list, eval_acc_list = [], []
    train_loss_list, validation_loss_list = [], []
    
    # Lists to store loss metrics at the end of every epoch
    train_loss_list_epochs, eval_loss_list_epochs = [], []

    # Also recording the training time
    start_time = time.time()
    for epoch in range(n_epochs):
        # Reset training counters for each epoch
        n_samples_train, n_correct_train = 0, 0
        loss_train = 0

        for step, (data, target) in enumerate(trainloader):
            # Set the model in training mode
            model.train()
            data, target = data.to(DEVICE), target.to(DEVICE)

            # Clear gradients from previous step
            optimizer.zero_grad()
            output = model(data)
            # Get the predicted class
            _, predicted = torch.max(output.data, 1)

            # Update training counters (number of samples processed and number of correct predictions)
            n_samples_train += target.size(0)
            n_correct_train += (predicted == target).sum().item()

            # Compute and accumulate loss
            loss = loss_fn(output, target)
            loss_train += loss.item() # Accumulate epoch loss for logging
            loss.backward()
            optimizer.step() # Update model parameters
            
            # Log training metrics every n_steps
            if (step + 1) % n_steps == 0:
                acc_train = 100.0 * n_correct_train / n_samples_train # Calculate training accuracy
                avg_loss_train = loss_train / (step + 1)  # Average loss up to current step
                
                # Store training loss and accuracy
                train_loss_list.append(avg_loss_train)
                train_acc_list.append(acc_train)
                print(f"Epoch [{epoch+1}/{n_epochs}], Step [{step+1}/{len(trainloader)}]")
                print(f"Training Accuracy: {acc_train:.2f}%")
                print(f"Training Loss: {avg_loss_train:.4f}")

        # Save the average validation loss for the epoch
        last_loss_train = loss_train / len(trainloader)  # Average loss for the epoch
        train_loss_list_epochs.append(last_loss_train)
        
        # Validation phase
        model.eval() # Set the model to evaluation mode
        n_samples_eval, n_correct_eval = 0, 0
        loss_eval = 0
        
        with torch.no_grad(): # Disable gradient computation for validation
            for step, (data, target) in enumerate(validloader):
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                # Get predicted class
                _, predicted = torch.max(output.data, 1)

                # Update evaluation counts
                n_samples_eval += target.size(0)
                n_correct_eval += (predicted == target).sum().item() # Count correct predictions
                
                # Compute and accumulate validation loss
                loss = loss_fn(output, target)
                loss_eval += loss.item()

                # Log validation metrics every n_steps
                if (step + 1) % n_steps == 0:
                    acc_eval = 100.0 * n_correct_eval / n_samples_eval # Calculate validation accuracy
                    avg_loss_eval = loss_eval / (step + 1)  # Average loss up to current step
                    # Store validation loss and accuracy
                    validation_loss_list.append(avg_loss_eval)
                    eval_acc_list.append(acc_eval)
                    print(f"Step [{step+1}/{len(validloader)}]")
                    print(f"Validation Accuracy: {acc_eval:.2f}%")
                    print(f"Validation Loss: {avg_loss_eval:.4f}")
            
            # Save the average validation loss for the epoch
            last_loss_eval = loss_eval / len(validloader)  # Average loss for the epoch
            eval_loss_list_epochs.append(last_loss_eval)
            # Calculate final validation accuracy for the epoch
            acc_eval = 100.0 * n_correct_eval / n_samples_eval
            print(f"Epoch [{epoch+1}/{n_epochs}], Validation Loss: {last_loss_eval:.4f}, Validation Accuracy: {acc_eval:.2f}%")


    # Test the model on the test dataset
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for data, target in testloader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = model(data)
            # Get predicted class
            _, predicted = torch.max(outputs.data, 1)
            n_samples += target.size(0)
            n_correct += (predicted == target).sum().item()

        # Calculate test accuracy
        acc = 100.0 * n_correct / n_samples
    end_time = time.time() # Record the end time of training
    print("Accuracy on the test set:", acc, "%")
    print(f"Training time: {end_time - start_time}")

    print(len(train_loss_list), len(validation_loss_list))
    print(len(train_loss_list_epochs), len(eval_loss_list_epochs))


    '''
    Q8
    '''
    # Plot training and validation loss over epochs
    plt.figure()
    plt.plot(range(n_epochs), train_loss_list_epochs)
    plt.plot(range(n_epochs), eval_loss_list_epochs)
    plt.legend(["Train loss", "Validation Loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss value")
    plt.show()

    '''
    Q9
    '''
    # Data augmentation transformations for training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(), # Horizontally flip the given image randomly with a default probability of 0.5
        transforms.RandomRotation(20), # Rotate the image by defined angle
        transforms.ToTensor(),
        transforms.Normalize(mean=0, std=1),
    ])

    # ReLoad the CIFAR-10 training dataset now with data augmentation
    dataset_train= datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    trainloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    # Reloading not also for test/validation -> data augmentation just on training set
    
    # Initialize a model with new architecture and define hyperparameters
    model = CNNGodzilla()
    learning_rate = 0.032
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' 
        if torch.backends.mps.is_available() else 'cpu')
    model = model.to(DEVICE)
    print("Working on", DEVICE)

    # Lists to store training and validation loss
    train_loss_list = []
    validation_loss_list = []
    
    # Thanks to the introductions of the Data Aug/Dropout who avoid overfitting I can add some epochs
    n_epochs = 6

    # Training
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
            
        # Compute and store training loss
        loss_train = loss_train / len(trainloader)
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

    # Test the model on the test dataset
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


    # Plot training and validation loss over epochs
    plt.figure()
    plt.plot(range(n_epochs), train_loss_list)
    plt.plot(range(n_epochs), validation_loss_list)
    plt.legend(["Train loss", "Validation Loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss value")
    plt.show()

    '''
    Q10 -  Code
    '''
    # Loop over a range of seeds to initialize random number generation for reproducibility
    for seed in range(5,10):
        torch.manual_seed(seed)
        print("Seed equal to ", torch.random.initial_seed())

        # Initialize the model, optimizer, loss function and LR
        model = CNNBasic()
        learning_rate = 0.029
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' 
            if torch.backends.mps.is_available() else 'cpu')
        model = model.to(DEVICE)
        print("Working on", DEVICE)

        # Initialize lists to track training and validation loss
        train_loss_list = []
        validation_loss_list = []
        n_epochs = 4

        # Training loop over epochs
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

            loss_train = loss_train / len(trainloader) #Compute training loss. 
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

        # Evaluate the model on the test set
        with torch.no_grad():
            # Initialize correct predictions counter and samples counter
            n_correct = 0
            n_samples = 0
            for data, target in testloader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1) # Get predicted classes
                n_samples += target.size(0)
                n_correct += (predicted == target).sum().item()

            # Calculate accuracy
            acc = 100.0 * n_correct / n_samples
        print("Accuracy on the test set:", acc, "%")
