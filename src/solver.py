import random
import numpy as np


from src.cube import Cube

class DeepCube:
    def autodidactic_iteration(self, N):
        cubes = [Cube(scrambled=True) for _ in range(N)]
