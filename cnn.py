import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from mask_dataset import MaskDataset
from PIL import Image

# Device configuration
# Have the GPU support
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-Parameters
# Can adjust epochs to train it better
num_epochs = 4
batch_size = 4
learning_rate = 0.001

# dataset has PILImage images of range [0,1]
# Resize the images to make sure every image is the same size
# Transform them to Tensors of normalized range [-1,1]
transform = transforms.Compose(
        [transforms.Resize((512,512)),
         transforms.ToTensor(),
         transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

# load the custom dataset with the parameters accordingly
dataset = MaskDataset("/root/project/train", "train_csv.csv", transform = transform)

# split the data into training data and testing data
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [92891,92892])

# these are the dataloaders
# define these to do automatic batch optimization and training
# could change up the parameters if needed
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

# define classes (this is only necessary for testing accuracy of each class)
classes = ('mask', 'no_mask')

# implement conv net
# must inherit nn.Module
# need to implement init and forward
class ConvNet(nn.Module):
    def __init__(self):
        # must do this:
        super(ConvNet, self).__init__()

        # this is for first layer - need to specify the parameters
        # input channel size is 3 bcs images have 3 color channels
        # output is 6 and kernel size is 5
        self.conv1 = nn.Conv2d(3, 6, 5)

        # pooling layer
        # kernel size 2 and stride 2
        # it is 2x2, so after each operation you shift it 2 pixels to the right that's why the stride is 2
        self.pool = nn.MaxPool2d(2,2)

        # second conv layer
        # now the input channel size must be the same as the last output channel size (6)
        # output 16 and kernel is still 5
        self.conv2 = nn.Conv2d(6,16,5)

        # fully connected layer
        # can try a diff int for the second no.
        # first is 125*125*16 (those were the dimensions given after the second pool) and MUST BE THAT
        #   now when we put to classification layers we want to flatten the 3d tensor to 1d which is why we made the
        #   input size of the first linear layer the 125*125*16
        self.fc1 = nn.Linear(125*125*16, 120)

        # next  fully connected layer
        # input is same as output from before
        self.fc2 = nn.Linear(120, 84)

        # final fully connected layer
        # now the input is 84
        # output is 2 bcs there are TWO different class
        # you can change the 120 and 84 but NOT the 2
        self.fc3 = nn.Linear(84, 2)

    # now apply layers in the forward pass
    def forward(self, x):
         # this is first convolutional layer and pool
         # relu applies the rectified linear unit function element-wise
         # apply activation function (doesn't change size) (this is the relu)
        x = self.pool(F.relu(self.conv1(x)))

        #pass to 2nd convoluted layer
        x = self.pool(F.relu(self.conv2(x)))

        # pass to the first fully connected layer
        # for this we have to flatten
        x = x.view(-1, 125*125*16) # pytorch automatically defines the correct size for us and -1 is the no  of batches we have
                            # 4 in this case, and then nxt MUST say 125*125*16
        # now call first fully connected layer
        # again apply activation function
        x = F.relu(self.fc1(x))
        # apply second
        x = F.relu(self.fc2(x))

        # at end last layer
        # NO activation function at end
        # NO softmax bcs the cross entropy loss has it
        x = self.fc3(x)

        return x

# create model
model = ConvNet().to(device)

# create loss and optimizer
# multiclass classification problem => cross entropy loss
# COULD CHANGE THIS 
criterion = nn.CrossEntropyLoss()
# use stochastic gradient descent to optimize model parameters
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

# training loop for batch optimization

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    # loop over training loader
    for i,(images, labels) in enumerate(train_loader):
        # origin shape: [4,3,1024,1024]
        # input_layer: 3 input channels, 6 output channels, 5 kernel size

        # push the images and image labels to device to get the GPU support
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        # must empty gradients first
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 2000 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

print ('Finished Training')

# evaluate the model
# wrap it in with torch.no_grad bcs we don't need backward propagation here
# basically calculating the accuracy here
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(2)]
    n_class_samples = [0 for i in range(2)]
    for images_test, labels_test in test_loader:
        images_test = images_test.to(device)
        labels_test = labels_test.to(device)
        outputs_test = model(images_test)
        # max returns (value, index)
        _, predicted = torch.max(outputs_test, 1)
        n_samples += labels_test.size(0)
        n_correct += (predicted ==labels_test).sum().item()

        for i in range(len(labels_test)):
            label = labels_test[i]
            pred = predicted[i]
            if(label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct/n_samples
    print(f'Accuracy of the network: {acc} %')

     #calculate accuracy for each single class
    for i in range(2):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')

torch.save(model.state_dict(),"/root/project/model.pth")