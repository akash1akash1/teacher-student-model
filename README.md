# Student teacher model

Report the performance of the student network and compare it with the teacher model. Also
compare the performance with and without EMA (Exponential Moving Average) updates.


## Deployment

below TeacherNet model has 12 convolutional layers with a kernel size of 3 and padding of 1.
The number of output channels for all layers is set to 12 and also includes max pooling layers with a kernel size of 2 and stride of 2. 
After the convolutional layers, the output is flattened and passed through a fully connected layer with 200 output neurons.


In forward method, the predicted outputs of the student model are normalized using the softmax function with a temperature parameter T, and then log-transformed using the LogSoftmax module of PyTorch. 
the true outputs of the teacher model are also normalized using the softmax function with the same temperature parameter T.

Finally, the KL divergence loss is computed between the log-transformed predicted outputs and the normalized true outputs using the nn.KLDivLoss

The code trains the network for a total of 10 epochs. The training and evaluation processes are repeated for each epoch, updating the network parameters to improve its performance on the training data and avoid overfitting.

initialize the running loss to 0.0. the input and label tensors are moved to the device (GPU or CPU), the optimizer's gradients are set to zero.
The teacher network's output tensor is detached from the computation graph, and the student network's output tensor is compared to the teacher's output tensor using cross-entropy loss function.loss is then backpropagated through the student network, and the optimizer updates the network's weights using the computed gradients. The running loss is updated with the loss value for each batch.
