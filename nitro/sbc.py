
from .mdl0 import Model
from enum import IntEnum


class SbcCmd(IntEnum):
    NOP = 0
    RET = 1
    NODE = 2
    MTX = 3
    MAT = 4
    SHP = 5
    NODEDESC = 6
    BB = 7
    BBY = 8
    NODEMIX = 9
    CALLDL = 10
    POSSCALE = 11
    ENVMAP = 12
    PRJMAP = 13

    CMD_MASK = 0x1F
    FLAG_MASK = 0xE0


class SbcInterpreter:
    def __init__(self, model: Model):
        pass
