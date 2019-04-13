from kociemba.cubiecube import CubieCube, moveCube
from kociemba.corner import URF, UFL, ULB, UBR, DFR, DLF, DBL, DRB
from kociemba.edge import UR, UF, UL, UB, DR, DF, DL, DB, FR, FL, BL, BR


class Cube(CubieCube):
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

    cpU = [UBR, URF, UFL, ULB, DFR, DLF, DBL, DRB]
    coU = [0, 0, 0, 0, 0, 0, 0, 0]
    epU = [UB, UR, UF, UL, DR, DF, DL, DB, FR, FL, BL, BR]
    eoU = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    cpR = [DFR, UFL, ULB, URF, DRB, DLF, DBL, UBR]
    coR = [2, 0, 0, 1, 1, 0, 0, 2]
    epR = [FR, UF, UL, UB, BR, DF, DL, DB, DR, FL, BL, UR]
    eoR = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    cpF = [UFL, DLF, ULB, UBR, URF, DFR, DBL, DRB]
    coF = [1, 2, 0, 0, 2, 1, 0, 0]
    epF = [UR, FL, UL, UB, DR, FR, DL, DB, UF, DF, BL, BR]
    eoF = [0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0]

    cpD = [URF, UFL, ULB, UBR, DLF, DBL, DRB, DFR]
    coD = [0, 0, 0, 0, 0, 0, 0, 0]
    epD = [UR, UF, UL, UB, DF, DL, DB, DR, FR, FL, BL, BR]
    eoD = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    cpL = [URF, ULB, DBL, UBR, DFR, UFL, DLF, DRB]
    coL = [0, 1, 2, 0, 0, 2, 1, 0]
    epL = [UR, UF, BL, UB, DR, DF, FL, DB, FR, UL, DL, BR]
    eoL = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    cpB = [URF, UFL, UBR, DRB, DFR, DLF, ULB, DBL]
    coB = [0, 0, 1, 2, 0, 0, 2, 1]
    epB = [UR, UF, UL, BR, DR, DF, DL, BL, FR, FL, UB, DB]
    eoB = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1]

    moves = {
        "U": CubieCube(cp=cpU, co=coU, ep=epU, eo=eoU),
        "R": CubieCube(cp=cpR, co=coR, ep=epR, eo=eoR),
        "F": CubieCube(cp=cpF, co=coF, ep=epF, eo=eoF),
        "D": CubieCube(cp=cpD, co=coD, ep=epD, eo=eoD),
        "L": CubieCube(cp=cpL, co=coL, ep=epL, eo=eoL),
        "B": CubieCube(cp=cpB, co=coB, ep=epB, eo=eoB),
    }

    @staticmethod
    def is_valid_move(move: str):
        if len(move) != 1 and len(move) != 2:
            return False
        return True

    def move_cube(self, move: str):
        """Moves one face of a Rubik's cube using standard notation"""
        if not self.is_valid_move(move):
            raise Exception("Move {} is not a valid move".format(move))
        elif len(move) == 1:
            self.multiply(self.moves[move])
        elif move[-1] == "2":
            move = move[0]
            self.multiply(self.moves[move]).multiply(self.moves[move])
        elif move[-1] =="'":
            move = move[0]
            self.multiply(self.moves[move]).multiply(self.moves[move]).multiply(self.moves[move])

    def is_solved(self):
        return self.toFaceCube().to_String() == "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"
