# -*- coding: utf-8 -*-
"""minor_2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lDBVumDREO_0WXJOHYetXxUSOdius7Xn
"""

!wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
!mkdir tiny-imagenet-200
!unzip -q tiny-imagenet-200.zip -d tiny-imagenet-200

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms



train_dir = "/content/tiny-imagenet-200/tiny-imagenet-200/train"
val_dir = "/content/tiny-imagenet-200/tiny-imagenet-200/val"


train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                       transforms.RandomRotation(10),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
val_transforms = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], 
                                                          [0.229, 0.224, 0.225])])

batch_size = 100

# Define data loaders
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)

import torch.nn as nn
import torch.nn.functional as F
class TeacherNet(nn.Module):
    def __init__(self):
        super(TeacherNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
        self.conv11 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(12 * 4 * 4, 200)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.pool(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.pool(x)
        x = x.view(-1, 12 * 4 * 4)
        x = self.fc(x)
        return x

import matplotlib.pyplot as plt

# Train the teacher network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_net = TeacherNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(teacher_net.parameters(), lr=0.001)

# Initialize lists to store the loss and accuracy values for each epoch
losses = []
accuracies = []

for epoch in range(10):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = teacher_net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss/len(train_loader)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

    # Evaluate the teacher network on the validation set
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = teacher_net(inputs)
            total += labels.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct/total * 100
    accuracies.append(accuracy)
    print(f"Accuracy on validation set: {accuracy:.2f}%")

# Plot the loss vs epoch graph
plt.plot(range(1, 11), losses)
plt.title("Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# Plot the accuracy vs epoch graph
plt.plot(range(1, 11), accuracies)
plt.title("Accuracy vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

import torch.nn.functional as F


class StudentNet(nn.Module):
    def __init__(self, T):
        super(StudentNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 200)
        self.T = T
        
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.fc2(x) / self.T
        return x

# import torch.nn as nn
# import torch.nn.functional as F
# class TeacherNet(nn.Module):
#     def __init__(self):
#         super(TeacherNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 12, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
#         self.conv4 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
#         self.conv5 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
#         self.conv6 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
#         self.conv7 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
#         self.conv8 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
#         self.conv9 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
#         self.conv10 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
#         self.conv11 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
#         self.conv12 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.fc = nn.Linear(12 * 4 * 4, 512)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.pool(x)
#         x = self.conv4(x)
#         x = self.conv5(x)
#         x = self.conv6(x)
#         x = self.pool(x)
#         x = self.conv7(x)
#         x = self.conv8(x)
#         x = self.conv9(x)
#         x = self.pool(x)
#         x = self.conv10(x)
#         x = self.conv11(x)
#         x = self.conv12(x)
#         x = self.pool(x)
#         x = x.view(-1, 12 * 4 * 4)
#         x = self.fc(x)
#         return x

class DistillKL(nn.Module):

  def __init__(self, T):

    super(DistillKL, self).__init__()
    self.T = T
    self.log_softmax = nn.LogSoftmax(dim=1)

  def forward(self, outputs, teacher_outputs):
    outputs = self.log_softmax(outputs / self.T)
    teacher_outputs = torch.softmax(teacher_outputs / self.T, dim=1)
    return nn.KLDivLoss(reduction='batchmean')(outputs, teacher_outputs) * self.T * self.T

import torch
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

T = 5
student_net = StudentNet(T=T).to(device)
criterion = DistillKL(T).to(device)

optimizer = optim.Adam(student_net.parameters(), lr=0.001)

loss_history = []
accuracy_history = []

for epoch in range(10):

    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):

        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = student_net(inputs)
        teacher_outputs = teacher_net(inputs).detach()

        loss = criterion(outputs, teacher_outputs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {running_loss/len(train_loader):.4f}")
    loss_history.append(running_loss/len(train_loader))

    student_net.eval()  # set the network to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = student_net(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct/total * 100
    print(f"Accuracy on validation set: {accuracy:.2f}%")
    accuracy_history.append(accuracy)

import matplotlib.pyplot as plt

# Plot loss vs epoch
plt.plot(range(1, 11), loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.show()

# Plot accuracy vs epoch
plt.plot(range(1, 11), accuracy_history)
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs Epoch')
plt.show()

import matplotlib.pyplot as plt

# define empty lists to store loss and accuracy
train_losses = []
val_losses = []
accuracy_list = []
alpha=0.8

# train the StudentNet model with EMA updates
num_epochs = 5

for epoch in range(num_epochs):
    train_loss = 0
    val_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        
        # compute the logits from the TeacherNet
        with torch.no_grad():
            teacher_logits = teacher_net(data)
        
        # compute the logits from the StudentNet
        student_logits = student_net(data)
        
        # compute the soft target loss
        loss_soft = criterion_soft(F.log_softmax(student_logits/T, dim=1),
                                   F.softmax(teacher_logits/T, dim=1)) * T * T
        
        # compute the EMA of the parameters
        for param_student, param_teacher in zip(student_net.parameters(), teacher_net.parameters()):
            if param_teacher.data.shape == param_student.data.shape:
                param_teacher.data.mul_(alpha).add_((1 - alpha) * param_student.data)
        
        loss_soft.backward()
        optimizer.step()
        
        # add the loss for this batch to the running train loss
        train_loss += loss_soft.item()
    student_net.eval()
    # evaluate the model on the validation set
    with torch.no_grad():
        for data, target in val_loader:
            outputs = student_net(data)
            val_loss += criterion(outputs, target).item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    # calculate the average train and validation loss for this epoch
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    
    # calculate the accuracy for this epoch
    accuracy = 100 * correct / total
    accuracy_list.append(accuracy)
    
    # append the train and validation loss to the lists
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print('Epoch: {}, Train Loss: {:.9f}, Validation Loss: {:.6f}, Validation Accuracy: {:.9f}%'.format(epoch+1, train_loss, val_loss, accuracy))

# plot the loss vs epoch graph
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend(frameon=False)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()