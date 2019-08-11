"""
Script for running DeepCube training
"""


import argparse

from deepcube import DeepCube


def get_parser():
    parser = argparse.ArgumentParser(description="Train a deep neural network to solve a Rubik's Cube.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--anew", action="store_true", help="Start training from a scratch.")


if __name__ == "__main__":
    deep_cube = DeepCube()
    deep_cube.learn(1)
    deep_cube.save_progress()