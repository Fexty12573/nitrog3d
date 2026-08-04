
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from . import matrix as mat
from . import mdl0
from .dl_encoder import DlEncoder
from .model import ImportedSubModel, ImportedMesh
from .quantize import box_exponent_for, pos_scale_for
from .sbc_encoder import SbcEncoder, preorder_bones, remap_bone_ids

MAX_NODES = 255
MAX_SHAPES = 255
MAX_MATERIALS = 255


class ModelBuilder:
    def __init__(self, sub: ImportedSubModel):
        self.sub = sub
        self.pos_scale = 1.0
        self.shapes: list[EmittedShape] = []
        self.dls: list[bytes] = []
        self.sbc = bytes()
        self.first_unused_mtx_stack_id = 0

    def build(self) -> mdl0.Model:
        # Step 1: Plan Nodes
        # Step 2: Plan Shapes
        # Step 3: Pos Scale
        # Step 4: Encode DLs
        # Step 5: Encode SBC
        # Step 6: Build Nodes
        # Step 7: Build Materials
        # Step 8: Build Shapes
        # Step 9: Build ModelInfo

        # Step 10: Build Model

        pass


@dataclass(slots=True)
class EmittedShape:
    index: int
    name: str
    mesh: ImportedMesh
    node: int  # remapped
    material: int
