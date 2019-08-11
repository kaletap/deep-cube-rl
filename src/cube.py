from __future__ import annotations
from typing import List
import random
import torch

UBL = 0
UBR = 1
UFL = 2
UFR = 3
DBL = 4
DBR = 5
DFL = 6
DFR = 7

UL = 0
UB = 1
UR = 2
UF = 3
DL = 4
DB = 5
DR = 6
DF = 7
FL = 8
FR = 9
BL = 10
BR = 11

"""
To every position of a corner we map which corner belongs here as well as it's orientation
    0: white/yellow side is facing up or down
    1: white/yellow side is facing front or back
    2: white/yellow side is facing right or left
"""
solved_corners = [
    (UBL, 0),
    (UBR, 0),
    (UFL, 0),
    (UFR, 0),
    (DBL, 0),
    (DBR, 0),
    (DFL, 0),
    (DFR, 0)
]

"""
Similarly for edges. True means that edge is oriented
"""
solved_edges = [
    (UL, True),
    (UB, True),
    (UR, True),
    (UF, True),
    (DL, True),
    (DB, True),
    (DR, True),
    (DF, True),
    (FL, True),
    (FR, True),
    (BL, True),
    (BR, True)
]

ACTIONS = ("U", "U'", "U2",
           "R", "R'", "R2",
           "F", "F'", "F2",
           "D", "D'", "D2",
           "L", "L'", "L2",
           "B", "B'", "B2",
           )


class Cube:
    """
    Class representing some state of a Rubik's Cube. Contains methods for moving each face and returning
    it's tensor representation.
    """

    def __init__(self, corners: List = None, edges: List = None, scramble_length: int = None):
        super().__init__()

        self.corners = corners or solved_corners.copy()
        self.edges = edges or solved_edges.copy()

        if scramble_length:
            self.scramble(scramble_length)

    def scramble(self, n_moves) -> None:
        sequence = random.choices(ACTIONS, k=n_moves)
        self.move(sequence)

    def __str__(self):
        return "Cube(corners={}, edges={})".format(self.corners, self.edges)

    def __repr__(self):
        return self.__str__()

    def _cycle_corners(self, corners_to_swap, orientation_map):
        last_corner, last_orientation = self.corners[corners_to_swap[-1]]
        for i in range(len(corners_to_swap) - 2, -1, -1):
            corner, orientation = self.corners[corners_to_swap[i]]
            self.corners[corners_to_swap[i + 1]] = corner, orientation_map[orientation]
        self.corners[corners_to_swap[0]] = last_corner, orientation_map[last_orientation]

    def _cycle_edges(self, edges_to_swap: List[int], change_orientation: bool) -> None:
        last_edge, last_orientation = self.edges[edges_to_swap[-1]]
        for i in range(len(edges_to_swap) - 2, -1, -1):
            edge, orientation = self.edges[edges_to_swap[i]]
            self.edges[edges_to_swap[i + 1]] = (edge, not orientation if change_orientation else orientation)
        self.edges[edges_to_swap[0]] = last_edge, (not last_edge if change_orientation else last_orientation)

    def _move_face(self, move) -> Cube:
        rl_map = r_map = {0: 1, 1: 0, 2: 2}
        ud_map = {0: 0, 1: 1, 2: 2}
        fb_map = {0: 2, 1: 1, 2: 0}
        if move == "R":
            self._cycle_corners([UFR, UBR, DBR, DFR], r_map)
            self._cycle_edges([UR, BR, DR, FR], False)
        elif move == "L":
            self._cycle_corners([UFL, DFL, DBL, UBL], rl_map)
            self._cycle_edges([UL, FL, DL, BL], False)
        elif move == "U":
            self._cycle_corners([UBL, UBR, UFR, UFL], ud_map)
            self._cycle_edges([UL, UB, UR, UF], False)
        elif move == "D":
            self._cycle_corners([DFL, DFR, DBR, DBL], ud_map)
            self._cycle_edges([DF, DR, DB, DL], False)
        elif move == "F":
            self._cycle_corners([UFL, UFR, DFR, DFL], fb_map)
            self._cycle_edges([UF, UR, DF, FL], True)
        elif move == "B":
            self._cycle_corners([UBR, UBL, DBL, DBR], fb_map)
            self._cycle_edges([UB, BL, DB, BR], True)
        return self

    @staticmethod
    def is_valid_move(move: str) -> bool:
        if len(move) != 1 and len(move) != 2:
            return False
        return True

    def move_single(self, move: str) -> None:
        if not self.is_valid_move(move):
            raise ValueError("Move {} is not a valid move".format(move))
        if len(move) == 2:
            if move[-1] == "'":
                self._move_face(move[0])._move_face(move[0])._move_face(move[0])
            elif move[-1] == "2":
                self._move_face(move[0])._move_face(move[0])
        else:
            self._move_face(move)

    def is_solved(self):
        return self.corners == solved_corners and self.edges == solved_edges

    def move(self, moves) -> None:
        if isinstance(moves, str):
            moves = moves.split(" ")
        for move in moves:
            self.move_single(move)

    def represent(self) -> torch.Tensor:
        """
        Gives representation of a cube as a vector to be fed into neural network
        """
        CORNERS_REPR_SIZE = 7 * 24
        corners_repr = torch.zeros(CORNERS_REPR_SIZE)
        for corner, (position, orientation) in enumerate(self.corners[:-1]):
            corners_repr[corner * 24 + position * 3 + orientation] = 1

        EDGES_REPR_SIZE = 11 * 24
        edges_repr = torch.zeros(EDGES_REPR_SIZE)
        for edge, (position, orientation) in enumerate(self.edges[:-1]):
            edges_repr[edge * 24 + position * 2 + orientation] = 1

        return torch.cat((corners_repr, edges_repr))
