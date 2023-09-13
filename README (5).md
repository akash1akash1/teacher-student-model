
# Student teacher model

Report the performance of the student network and compare it with the teacher model. Also
compare the performance with and without EMA (Exponential Moving Average) updates.


## Deployment

below TeacherNet model has 12 convolutional layers with a kernel size of 3 and padding of 1. The number of output channels for all layers is set to 12 and also includes max pooling layers with a kernel size of 2 and stride of 2. After the convolutional layers, the output is flattened and passed through a fully connected layer with 200 output neurons.

```bash

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


```


In forward method, the predicted outputs of the student model are normalized using the softmax function with a temperature parameter T, and then log-transformed using the LogSoftmax module of PyTorch. 
the true outputs of the teacher model are also normalized using the softmax function with the same temperature parameter T.

Finally, the KL divergence loss is computed between the log-transformed predicted outputs and the normalized true outputs using the nn.KLDivLoss


```bash

class DistillKL(nn.Module):

  def __init__(self, T):

    super(DistillKL, self).__init__()
    self.T = T
    self.log_softmax = nn.LogSoftmax(dim=1)

  def forward(self, outputs, teacher_outputs):
    outputs = self.log_softmax(outputs / self.T)
    teacher_outputs = torch.softmax(teacher_outputs / self.T, dim=1)
    return nn.KLDivLoss(reduction='batchmean')(outputs, teacher_outputs) * self.T * self.T



```

 The code trains the network for a total of 10 epochs. The training and evaluation processes are repeated for each epoch, updating the network parameters to improve its performance on the training data and avoid overfitting.

```bash

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

```
initialize the running loss to 0.0. the input and label tensors are moved to the device (GPU or CPU), the optimizer's gradients are set to zero.
The teacher network's output tensor is detached from the computation graph, and the student network's output tensor is compared to the teacher's output tensor using cross-entropy loss function.loss is then backpropagated through the student network, and the optimizer updates the network's weights using the computed gradients. The running loss is updated with the loss value for each batch.


```bash

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



```

```bash

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



```