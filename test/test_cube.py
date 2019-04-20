from src.cube import Cube, solved_corners, solved_edges

import torch


def test_single_move():
    cube = Cube()
    cube.move_single("L")
    assert(not cube.is_solved())
    cube.move_single("L'")
    assert(cube.is_solved())

def test_triple_move():
    cube = Cube()
    cube.move_single("R")
    cube.move_single("R")
    cube.move_single("R")
    cube.move_single("R")
    assert(cube.is_solved())

def test_reverse():
    cube = Cube()
    sequence = "R U R' U R U2 R'"
    for move in sequence.split(" "):
        cube.move_single(move)
    assert(not cube.is_solved())
    reverse_sequence = "R U2 R' U' R U' R'"
    for move in reverse_sequence.split(" "):
        cube.move_single(move)
    assert(cube.is_solved())

def test_move():
    cube = Cube()
    sequence = "R U R' U R U2 R'"
    reverse_sequence = "R U2 R' U' R U' R'"
    cube.move(sequence)
    assert(not cube.is_solved())
    cube.move(reverse_sequence)
    assert(cube.is_solved())

def test_repr():
    cube = Cube()
    expected_repr = torch.zeros(432)
    for i in range(7):
        expected_repr[i*24 + i*3] = 1
    for i in range(11):
        expected_repr[168 + i*24 + i*2 + 1] = 1
    assert(all(cube.represent() == expected_repr))


