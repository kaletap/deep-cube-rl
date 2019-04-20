import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.cube import Cube


INPUT_DIM = 7*24 + 11*24
ACTION_DIM = 12
OUTPUT_DIM = ACTION_DIM + 1

LAYER_1_DIM = 4096
LAYER_2_DIM = 2048
ACTION_LAYER_DIM = 512
VALUE_LAYER_DIM = 512


class RubikNet(nn.Module):
    def __init__(self):
        super(RubikNet, self).__init__()
        self.first = nn.Linear(INPUT_DIM, LAYER_1_DIM)
        self.second = nn.Linear(LAYER_1_DIM, LAYER_2_DIM)
        self.action_layer_1 = nn.Linear(LAYER_2_DIM, ACTION_LAYER_DIM)
        self.value_layer_1 = nn.Linear(LAYER_2_DIM, VALUE_LAYER_DIM)
        self.value_layer2 = nn.Linear(VALUE_LAYER_DIM, 1)

    def forward(self, x):
        x = self.first(x)  # 4096
        x = self.second(F.elu(x))  # 2048
        x_actions = self.action_layer_1(F.elu(x))
        x_actions = F.softmax(x_actions, ACTION_DIM)
        x_value = self.value_layer_1(F.elu(x))
        x_value = self.value_layer_2(F.elu(x_value))
        return x_actions, x_value

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def customized_loss(y_action_pred, y_value_pred, y_action, y_value, alpha=1):
    action_criterion = nn.CrossEntropyLoss()
    action_loss = action_criterion(y_action_pred, y_action)
    value_criterion = nn.SmoothL1Loss()
    value_loss = value_criterion(y_value_pred, y_value)
    return action_loss + alpha*value_loss


net = RubikNet()

x = torch.zeros(432)
for i in range(7):
    x[i*24 + i*3] = 1
for i in range(11):
    x[168 + i*24 + i*2 + 1] = 1

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

trainloader = [x, [1]]  # TODO: implement trainloader

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, action_labels, value_label = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        action_output, value_output = net(inputs)
        loss = customized_loss(action_output, value_output, action_labels, value_label)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

