from src.cube import Cube, solved_corners, solved_edges

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
