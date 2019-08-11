import torch.nn as nn
import torch.nn.functional as F


ACTIONS = ("U", "U'", "U2",
    "R", "R'", "R2",
    "F", "F'", "F2",
    "D", "D'", "D2",
    "L", "L'", "L2",
    "B", "B'", "B2",
           )

INPUT_DIM = 7*24 + 11*24
ACTION_DIM = len(ACTIONS)
OUTPUT_DIM = ACTION_DIM + 1  # one dimension for value function

LAYER_1_DIM = 4096
LAYER_2_DIM = 2048
ACTION_LAYER_DIM = 512
VALUE_LAYER_DIM = 512


class RubikNet(nn.Module):
    """
    Definition of Neural Network we want to train. For now architecture is identical to the original.
    We use same weights for value function and actions.
    """
    def __init__(self):
        super(RubikNet, self).__init__()
        self.first = nn.Linear(INPUT_DIM, LAYER_1_DIM)
        self.second = nn.Linear(LAYER_1_DIM, LAYER_2_DIM)
        self.action_layer_1 = nn.Linear(LAYER_2_DIM, ACTION_LAYER_DIM)
        self.action_layer_2 = nn.Linear(ACTION_LAYER_DIM, ACTION_DIM)
        self.value_layer_1 = nn.Linear(LAYER_2_DIM, VALUE_LAYER_DIM)
        self.value_layer_2 = nn.Linear(VALUE_LAYER_DIM, 1)

    def forward(self, x):
        x = self.first(x)  # 4096
        x = self.second(F.elu(x))  # 2048
        x_actions = self.action_layer_1(F.elu(x))  # 512
        x_actions = self.action_layer_2(F.elu(x_actions))  # 18
        x_actions = F.softmax(x_actions)  # normalized
        x_value = self.value_layer_1(F.elu(x))
        x_value = self.value_layer_2(F.elu(x_value))
        return x_actions, x_value
