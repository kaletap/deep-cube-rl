from src.cube import Cube

def test_one_move():
    cube = Cube()
    cube.move_cube("R")
    cube.move_cube("R'")
    assert(cube.is_solved())

def test_permutation():
    cube = Cube()
    sequence = "R U R' U R U2 R'"
    reverse_sequence = "R U2 R' U' R U' R'"
    for move in sequence.split(" "):
        cube.move_cube(move)
    assert(not cube.is_solved())
    for move in reverse_sequence.split(" "):
        cube.move_cube(move)
    assert(cube.is_solved())
