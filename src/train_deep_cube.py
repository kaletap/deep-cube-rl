"""
Script for running DeepCube training
"""

import argparse
from tqdm import tqdm

from deepcube import DeepCube


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a deep neural network to solve a Rubik's Cube.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--anew", action="store_true", help="Start training from a scratch.")
    return parser


if __name__ == "__main__":
    deep_cube = DeepCube(restore=True, verbose=True)
    for _ in tqdm(range(20)):
        for scramble_length in range(1, 6):
            print("Currently learning on cubes scrambled with {} moves".format(scramble_length))
            deep_cube.adi(4000, scramble_length)

    deep_cube.save_progress()

