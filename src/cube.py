import random
import numpy as np

UFR = 0
UFL = 1
UBL = 2
UBR = 3
DFR = 4
DFL = 5
DBL = 6
DBR = 7

UR = 0
UF = 1
UL = 2
UB = 3
DR = 4
DF = 5
DL = 6
DB = 7
FR = 8
FL = 9
BL = 10
BR = 11

"""
To every position of a corner we map which corner belongs here as well as it's orientation
    0: white/yellow side is facing up or down
    1: white/yellow side is facing front or back
    2: white/yellow side is facing right or left
"""
solved_corners = {
            UBL: (UBL, 0),
            UBR: (UBR, 0),
            UFL: (UFL, 0),
            UFR: (UFR, 0),
            DBL: (DBL, 0),
            DBR: (DBR, 0),
            DFL: (DFL, 0),
            DFR: (DFR, 0)
        }

"""
Similarly for edges. True means that edge is oriented
"""
solved_edges = {
            UL: (UL, True),
            UB: (UB, True),
            UR: (UR, True),
            UF: (UF, True),
            DL: (DL, True),
            DB: (DB, True),
            DR: (DR, True),
            DF: (DF, True),
            FL: (FL, True),
            FR: (FR, True),
            BL: (BL, True),
            BR: (BR, True)
        }

class Cube():
    def __init__(self, scrambled=False):
        super().__init__()

        self.corners = solved_corners.copy()
        self.edges = solved_edges.copy()

        if scrambled:
            self.scramble()

    def _cycle_corners(self, corners_to_swap, orientation_map):
        last_corner, last_orientation = self.corners[corners_to_swap[-1]]
        for i in range(len(corners_to_swap) - 2, -1, -1):
            corner, orientation = self.corners[corners_to_swap[i]]
            print(i, corner)
            self.corners[corners_to_swap[i+1]] = corner, orientation_map[orientation]
        self.corners[corners_to_swap[0]] = last_corner, orientation_map[last_orientation]

    def _cycle_edges(self, edges_to_swap, change_orientation: bool):
        last_edge, last_orientation = self.edges[edges_to_swap[-1]]
        for i in range(len(edges_to_swap) - 2, -1, -1):
            edge, orientation = self.edges[edges_to_swap[i]]
            self.edges[edges_to_swap[i + 1]] = (edge, not orientation if change_orientation else orientation)
        self.edges[edges_to_swap[0]] = last_edge, (not last_edge if change_orientation else last_orientation)

    def _move_face(self, move):
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
    def is_valid_move(move: str):
        if len(move) != 1 and len(move) != 2:
            return False
        return True

    def move_single(self, move: str):
        if not self.is_valid_move(move):
            raise ValueError("Move {} is not a valid move".format(move))
        if len(move) == 2:
            if move[-1] == "'":
                self._move_face(move[0])._move_face(move[0])._move_face(move[0])
            elif move[-1] == "2":
                self._move_face(move[0])._move_face(move[0])
        else:
            self._move_face(move)

    def __str__(self):
        cube_string = self.toFaceCube().to_String()
        return "-".join([cube_string[9*i : 9*i+9] for i in range(6)])

    actions = ["U", "U'", "U2",
               "R", "R'", "R2",
               "F", "F'", "F2",
               "D", "D'", "D2",
               "L", "L'", "L2",
               "B", "B'", "B2",
               ]

    def is_solved(self):
        return self.corners == solved_corners and self.edges == solved_edges

    def move(self, moves):
        if isinstance(moves, str):
            moves = moves.split(" ")
        for move in moves:
            self.move_face(move)

    def scramble(self, n_moves):
        sequence = random.choices(self.actions, k=n_moves)
        self.move(sequence)

    def represent(self):
        """Gives representation of a cube as a vector"""
