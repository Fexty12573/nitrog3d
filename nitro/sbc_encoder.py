from __future__ import annotations
from dataclasses import dataclass
from .model import ImportedSubModel, ImportedMesh, Bone
from .binary import BinaryWriter
from .sbc import NodeDescFlag
from .sbc_commands import *
from .tex0 import TexGen
import numpy as np
from collections import defaultdict

_TEXGEN_CMDS: dict[TexGen, type[SbcEnvMap | SbcPrjMap]] = {
    TexGen.NORMAL: SbcEnvMap,
    TexGen.VERTEX: SbcPrjMap,
}

# Usually 31 but G3D reserves 2 for its own purposes, so we only get to use 29
MAX_MTX_SLOTS = 29


class SbcEncoder:
    def __init__(self, model: ImportedSubModel, mapping: BoneMapping, shape_map: dict[str, int]):
        self.model = model
        self.bones = mapping.bones
        self.id_map = mapping.id_map
        self.id_map[-1] = 0
        self.shape_map = shape_map
        self.nodes: dict[int, Node] = {}
        self.cmds: list[SbcCommand] = []
        self.sbc = BinaryWriter()
        self.next_mtx_stack_id = 0
        self.current_bound_mat = -1
        self.live_after: list[bool] = []
        self.dies_at: list[list[int]] = []

        assert all(len(set(m.vertex_bone)) ==
                   1 for m in model.meshes), "multi matrix shapes not supported yet"
        mesh_nodes = [(m, m.vertex_bone[0]) for m in model.meshes]

        self.node_to_mesh: defaultdict[int, list[ImportedMesh]] \
            = defaultdict(list)
        for m, n in mesh_nodes:
            self.node_to_mesh[self.id_map[n]].append(m)

    def encode(self) -> bytes:
        self._phase1()
        self._phase2()
        self._phase3()

        for cmd in self.cmds:
            self.sbc.write_bytes(cmd.to_bytes())

        return self.sbc.get_bytes()

    def _phase1(self):
        # Phase 1: Emit the command stream, but don't care about matrix stack slots.
        # Draws are interleaved with NODEDESCs to reduce matrix stack usage.

        for id, bone in self.bones:
            node = Node(
                id=self.id_map[id],
                parent=self.id_map[bone.parent] if bone.parent != id else self.id_map[id],
                world_mtx=bone.world_mtx,
                desc_idx=len(self.nodes),
                ssc=bone.scale_compensate
            )

            self.nodes[node.id] = node
            self._emit_nodedesc(node)

            meshes = self.node_to_mesh.get(node.id)
            if meshes:
                self._emit_draw_group(node.id, meshes)

        self._emit_ret()

    def _emit_draw_group(self, node: int, meshes: list[ImportedMesh]):
        # All meshes bound to this node share its visibility
        self._emit_node(node, meshes[0].visible)

        self._emit_posscale()
        for mesh in meshes:
            if self.current_bound_mat != mesh.material:
                self._emit_mat(mesh.material)
                self.current_bound_mat = mesh.material

            self._emit_shp(self.shape_map[mesh.name])
        self._emit_posscale(inverse=True)

    def _phase2(self):
        # Phase 2: Compute liveness of bound matrices and when they die, so we can
        # more efficiently allocate matrix stack slots in phase 3.
        n = len(self.cmds)
        self.live_after = [False] * n
        self.dies_at = [[] for _ in range(n)]

        seen: set[int] = set()
        for i in range(n - 1, -1, -1):
            cmd = self.cmds[i]
            bound = cmd.binds()
            self.live_after[i] = bound is not None and bound in seen
            for node in cmd.reads():
                if node not in seen:
                    seen.add(node)
                    self.dies_at[i].append(node)

    def _phase3(self):
        # Phase 3: Allocate matrix stack slots
        slots: list[int | None] = [None] * MAX_MTX_SLOTS
        where: dict[int, int] = {}
        current: int | None = None
        highest_slot = 0

        for i, cmd in enumerate(self.cmds):
            need = cmd.needs()
            if need is not None and need != current:
                if need not in where:
                    raise ValueError(
                        f"node {need}'s matrix is needed at command {i} but was never stored"
                    )
                cmd.set_restore(where[need])
                current = need

            bound = cmd.binds()
            if bound is not None:
                current = bound
                if self.live_after[i] and bound not in where:
                    slot = _lowest_free(slots)
                    slots[slot] = bound
                    where[bound] = slot
                    cmd.store(slot)
                    highest_slot = max(highest_slot, slot + 1)

            # Check for any matrices that die at this command and free their slots for reuse
            for dead in self.dies_at[i]:
                slot = where.pop(dead, None)
                if slot is not None:
                    slots[slot] = None

        self.next_mtx_stack_id = highest_slot

    def _emit_nodedesc(self, node: Node):
        self.cmds.append(SbcNodeDesc(
            node=node.id,
            parent=node.parent,
            flags=NodeDescFlag.SSC_APPLY if node.ssc else NodeDescFlag.NONE
        ))

        # If this node has SSC, we need to adjust all parents up the hierarchy
        if node.ssc:
            parent_id = node.parent
            while parent_id != node.id:
                parent_node = self.nodes[parent_id]
                self.cmds[parent_node.desc_idx].ssc_parent()
                parent_id = parent_node.parent

    def _emit_node(self, node: int, visible: bool):
        self.cmds.append(SbcNode(node=node, visible=visible))

    def _emit_mtx(self, node: int):
        self.cmds.append(SbcMtx(node=node))

    def _emit_posscale(self, inverse: bool = False):
        self.cmds.append(SbcPosScale(inverse=inverse))

    def _emit_mat(self, idx: int):
        self.cmds.append(SbcMat(idx=idx))
        self._emit_texgen(idx)

    def _emit_texgen(self, idx: int):
        if idx >= len(self.model.materials):
            return

        param = self.model.materials[idx].tex_img_param
        cmd = _TEXGEN_CMDS.get(param.texgen) if param else None
        if cmd is None:
            return

        self.cmds.append(cmd(mat=idx))

    def _emit_shp(self, idx: int):
        self.cmds.append(SbcShp(idx=idx))

    def _emit_ret(self):
        self.cmds.append(SbcRet())

    def _emit_nop(self):
        self.cmds.append(SbcNop())


def _lowest_free(slots: list[int | None]) -> int:
    for i, occupant in enumerate(slots):
        if occupant is None:
            return i
    raise ValueError(
        f"Ran out of matrix stack slots. The model is too complex"
    )


@dataclass(slots=True)
class Node:
    id: int
    parent: int
    world_mtx: np.ndarray
    desc_idx: int  # Index of this node's NODEDESC command
    ssc: bool = False


@dataclass(slots=True)
class BoneMapping:
    bones: list[tuple[int, Bone]]
    id_map: dict[int, int]
