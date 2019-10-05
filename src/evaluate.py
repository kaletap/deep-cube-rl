"""
Script for evaluating DeepCube
"""

from cube import Cube
from deepcube import DeepCube



if __name__ == "__main__":
    cube = Cube()
    cube.move("R'")
    deep_cube = DeepCube(restore=True)
    solution = deep_cube.solve(cube)
    print(solution)
