import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.cube import Cube

ACTIONS = ("U", "U'", "U2",
    "R", "R'", "R2",
    "F", "F'", "F2",
    "D", "D'", "D2",
    "L", "L'", "L2",
    "B", "B'", "B2",
           )

INPUT_DIM = 7*24 + 11*24
ACTION_DIM = len(ACTIONS)
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


class DeepCube:
    def __init__(self):
        self.net = RubikNet()
        self.actions = ACTIONS

    def trainloader(self):
        return [0, 1]

    def train(self, trainloader, epochs=2):
        def customized_loss(y_action_pred, y_value_pred, y_action, y_value, alpha=1):
            action_criterion = nn.CrossEntropyLoss()
            action_loss = action_criterion(y_action_pred, y_action)
            value_criterion = nn.SmoothL1Loss()
            value_loss = value_criterion(y_value_pred, y_value)
            return action_loss + alpha * value_loss

        optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs
                inputs, action_labels, value_label = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                action_output, value_output = self.net(inputs)
                loss = customized_loss(action_output, value_output, action_labels, value_label)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        print('Finished Training')

    def adi(self, n, scramble_length):
        """Autodidactic Iteration described in paper "Solving the Rubik's Cube Without Human Knowledge (2018)"""
        cubes = [Cube(scramble_length=scramble_length) for _ in range(n)]
        target_probas = list()
        target_values = list()

        for cube in cubes:
            values = dict()
            for a in self.actions:
                new_cube = Cube(cube.corners, cube.edges)
                new_cube.move_single(a)
                y_actions, y_value = self.net(new_cube.represent())
                values[a] = y_value + (1 if new_cube.is_solved() else -1)
            best_action = max(values.iterkeys(), key=(lambda key: values[key]))
            # we set target probas to distribution with all mass in estimated best action
            y_probas = list(map(lambda ac: 1 if ac==best_action else 0, self.actions))
            y_value = values[best_action]

            target_probas.append(y_probas)
            target_values.append(y_value)

        trainloader = ([cube.represent() for cube in cubes], target_probas, target_values)

        self.train(trainloader)

    def learn(self, iterations_per_scramble_length):
        for i in range(1, 20):
            print("Currently learning on cubes scrmabled with {} moves".format(i))
            self.adi(iterations_per_scramble_length, i)
