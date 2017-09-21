import math
import torch
import numpy as np
import torchvision.datasets as dsets
import torch.optim as optim
from neural_network import NeuralNetwork
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F


# Hyper Parameters
D_in, H, D_out, mini_batch_size = 784, 200, 10, 50
learning_rate = 1e-3
max_iter = 50
momentum = 0.5

# MNIST Dataset
train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=False)
test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=mini_batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=mini_batch_size, shuffle=False)

model = torch.nn.Sequential(torch.nn.Linear(D_in, H), torch.nn.Sigmoid(), torch.nn.Linear(H, D_out))
loss_fn = torch.nn.MSELoss(size_average=False)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

for epoch in range(0, max_iter):
    for batch_idx, (data, target) in enumerate(train_loader):
        # data = data.numpy()
        # Reshape or flatten the input batch
        data = data.view(mini_batch_size, 1, 784)
        target_onehot = target.numpy()
        target_onehot = (np.arange(D_out) == target_onehot[:, None]).astype(np.float32)
        target_onehot = torch.from_numpy(target_onehot)
        data, target = Variable(data), Variable(target_onehot)
        optimizer.zero_grad()
        # print(data.size())
        output = model(data)

        # print(output.view(mini_batch_size))
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.data[0]))
