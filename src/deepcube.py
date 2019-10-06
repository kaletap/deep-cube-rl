from typing import List, Optional
import logging
import os
import time
import datetime
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim

from cube import Cube
from rubiknet import RubikNet, ACTIONS, ACTION_DIM


logger = logging.getLogger()

SAVE_PATH_ROOT = "weights"
NET_STATE_FILE = "rubik_net_state_dict.pickle"


class DeepCube:
    """
    Class responsible for actual neural network training on random Rubik's Cubes (Autodidactic Iteration)
    """
    def __init__(self, restore: bool = False, verbose=True):
        if verbose:
            self.verbose = verbose
            logging.basicConfig(level=logging.DEBUG)
        self.actions = ACTIONS
        self.net = RubikNet()
        if restore:
            def is_date_folder(folder_name: str):
                # folder_name has to have 19 characters and not consist of any letter
                return len(folder_name) == 19 and all(not c.isalpha() for c in folder_name)
            dates = [folder for folder in os.listdir(SAVE_PATH_ROOT) if is_date_folder(folder)]
            if len(dates) == 0:
                print("No folder to restore from!")
                return
            path = os.path.join(SAVE_PATH_ROOT, max(dates), NET_STATE_FILE)
            self.net.load_state_dict(torch.load(path))
            print("Restored weights from {}".format(path))

    def train(self, trainloader: list, weight: float, epochs: int = 2) -> None:
        logger.debug("DeepCube.train(weight={}, epochs={})".format(weight, epochs))

        def customized_loss(y_action_pred, y_value_pred, y_action, y_value, weight, alpha=1):
            action_criterion = nn.CrossEntropyLoss()
            action_loss = action_criterion(y_action_pred, y_action)
            value_criterion = nn.SmoothL1Loss()
            value_loss = value_criterion(y_value_pred, y_value)
            return weight * (action_loss + alpha * value_loss)

        optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(trainloader):
                # get the inputs
                inputs, action_labels, value_label = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                action_output, value_output = self.net(inputs)
                loss = customized_loss(action_output, value_output, action_labels, value_label, weight=weight)
                loss.backward(retain_graph=True)
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        print('Finished Training')

    def adi(self, n_cubes: int, scramble_length: int) -> None:
        """
        Autodidactic Iteration described in paper "Solving the Rubik's Cube Without Human Knowledge" (2018).
        """

        logger.debug("DeepCube.adi(n_cubes={}, scramble_length={})".format(n_cubes, scramble_length))

        cubes = [Cube(scramble_length=scramble_length) for _ in range(n_cubes)]
        logger.debug("Finished creating cubes")
        target_probas = list()
        target_values = list()

        log_every = 100
        cube_processed = 0
        for cube in cubes:
            start_time = time.time()
            values = dict()
            for i, a in enumerate(self.actions):
                new_cube = Cube(cube.corners, cube.edges)
                new_cube.move_single(a)
                y_actions, y_value = self.net(new_cube.represent())
                values[i] = y_value + (1 if new_cube.is_solved() else -1)
            best_action_index = max(values, key=(lambda key: values[key]))
            # we set target probas to distribution with all mass in estimated best action
            y_probas = best_action_index
            y_value = values[best_action_index]
            target_probas.append(y_probas)
            target_values.append(y_value)

            cube_processed += 1
            if cube_processed % log_every == 0:
                logger.debug("Time for processing {} cubes: {}".format(log_every, time.time() - start_time))

        logger.debug("Finished each cube")
        input_ = torch.stack(tuple(cube.represent() for cube in cubes))
        logger.debug("Obtained representation for each input cube")
        target_probas = torch.Tensor(target_probas).long()
        target_values = torch.stack(target_values)

        trainloader = [[input_, target_probas, target_values]]

        # higher training weight to cubes closed to solved (due to divergent solutions otherwise)
        self.train(trainloader, weight=1/scramble_length)

    def learn(self, iterations_per_scramble_length: int) -> None:
        """
        Main interface for learning a model.
        """
        for i in range(1, 10):
            print("Currently learning on cubes scrambled with {} moves".format(i))
            self.adi(iterations_per_scramble_length, i)

    def save_progress(self, path: str = None) -> None:
        """
        :param path: Path in which weights are going to be located
        :return:
        """
        folder_name = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        folder_path = path or os.path.join(SAVE_PATH_ROOT, folder_name)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        path = os.path.join(folder_path, NET_STATE_FILE)
        torch.save(self.net.state_dict(), path)
        # TODO: write some metadata as well
        # metadata_path = os.path.join(folder_path, "info.json")

    # TODO: Monte Carlo implementation
    def solve(self, cube: Cube, move_limit: int = 200, n_tries: int = 500) -> Optional[List[str]]:
        """
        Naive implementation of solving strategy - sampling from move probability.
        """
        best_solution = None
        for i in range(n_tries):
            print(i)
            solution = []
            _cube = Cube(cube.corners, cube.edges)
            while not _cube.is_solved():
                probas = self.net(_cube.represent())[:ACTION_DIM][0]
                # best_move = ACTIONS[torch.argmax(probas)]  # greedy strategy
                best_move = np.random.choice(ACTIONS, p=probas.detach().numpy())
                _cube.move(best_move)
                solution.append(best_move)
                if len(solution) >= move_limit:
                    if self.verbose:
                        print(solution[:5], str(probas.detach().numpy()))
                    break
            if best_solution and len(solution) < len(best_solution):
                best_solution = solution
                if len(best_solution) < 20:
                    return best_solution

        return best_solution if len(best_solution) < move_limit else None
