from __future__ import annotations
from dataclasses import dataclass
from .model import ImportedSubModel, ImportedMesh, Bone
from .binary import BinaryWriter
from .sbc import SbcCmd, SbcOpt
from . import matrix as mat
import numpy as np
from collections import Counter, defaultdict


class SbcEncoder:
    def __init__(self, model: ImportedSubModel):
        self.model = model
        self.bones = _preorder_bones(model.bones)
        self.id_map = _remap_bone_ids(self.bones)
        self.id_map[-1] = 0
        self.nodes: list[Node] = []
        self.sbc = BinaryWriter()
        self.next_mtx_stack_id = 0
        self.last_desc: Node | None = None
        self.last_material: int | None = None
        self.current_bound_node = -1
        self.current_bound_mat = -1

        assert all(model.meshes, lambda m: set(m.vertex_bone) ==
                   1), "multi matrix shapes not supported yet"
        mesh_nodes = [(m, m.vertex_bone[0]) for m in model.meshes]

        self.node_to_mesh: defaultdict[int, list[ImportedMesh]] \
            = defaultdict(list)
        for m, n in mesh_nodes:
            self.node_to_mesh[self.id_map[n]].append(m)

        # TODO: Use for "is STORE necessary" checks
        parents = Counter([b.parent for id, b in self.bones if b.parent != id])
        self.branches: set[int] = {k for k, v in dict(parents) if v > 1}

    def encode(self) -> bytes:
        # Step 1: NODEDESCs
        for id, bone in self.bones:
            node = Node(
                id=self.id_map[id],
                parent=self.id_map[bone.parent] if bone.parent != id else self.id_map[id],
                world_mtx=bone.world_mtx,
                mtx_slot=self.next_mtx_stack_id
            )
            self.next_mtx_stack_id += 1

            self.nodes.append(node)
            self._emit_nodedesc(node)
            self.last_desc = node

        # Step 2: Draw Shapes
        for bound_node, meshes in self.node_to_mesh.items():
            if self.current_bound_node != bound_node:
                slot = self.nodes[bound_node].mtx_slot
                self._emit_mtx(slot)
                self.current_bound_node = bound_node

            self._emit_posscale()

            for i, mesh in enumerate(meshes):
                if self.current_bound_mat != mesh.material:
                    self._emit_mat(mesh.material)
                    self.current_bound_mat = mesh.material

                self._emit_shp(i)

        self._emit_posscale(True)

    def _emit_nodedesc(self, node: Node):
        # local = mat.inverse(self.model.bones[bone.parent].world_mtx) @ bone.world_mtx

        cmd = _make_sbc_cmd(SbcCmd.NODEDESC, SbcOpt.STORE)

        # If the last node is NOT our parent, we need to restore the actual parent node's matrix slot
        restore = self.last_desc is not None and self.last_desc.id != node.parent
        if restore:
            cmd |= SbcOpt.RESTORE

        self.sbc.write_u8(int(cmd))
        self.sbc.write_u8(node.id)
        self.sbc.write_u8(node.parent)
        self.sbc.write_u8(0)  # TODO: flags
        self.sbc.write_u8(node.mtx_slot)
        if restore:
            self.sbc.write_u8(self.nodes[node.parent].mtx_slot)

    def _emit_mtx(self, slot: int):
        cmd = _make_sbc_cmd(SbcCmd.MTX, SbcOpt.NONE)
        self.sbc.write_u8(int(cmd))
        self.sbc.write_u8(slot)

    def _emit_posscale(self, inverse: bool = False):
        cmd = _make_sbc_cmd(
            SbcCmd.POSSCALE, SbcOpt.INVERSE if inverse else SbcOpt.NONE)
        self.sbc.write_u8(int(cmd))

    def _emit_mat(self, idx: int):
        cmd = _make_sbc_cmd(SbcCmd.MAT, SbcOpt.NONE)
        self.sbc.write_u8(int(cmd))
        self.sbc.write_u8(idx)

    def _emit_shp(self, idx: int):
        cmd = _make_sbc_cmd(SbcCmd.SHP, SbcOpt.NONE)
        self.sbc.write_u8(int(cmd))
        self.sbc.write_u8(idx)


def _preorder_bones(bones: list[Bone]) -> list[tuple[int, Bone]]:
    root_id, root = next(
        filter(lambda b: b[0] == b[1].parent, enumerate(bones)), (0, bones[0])
    )
    return _get_children(root_id, root, bones)


def _get_children(id: int, bone: Bone, bones: list[Bone]) -> list[tuple[int, Bone]]:
    children: list[Bone] = []
    for (cid, child) in filter(lambda bone: bone[1].parent == id and bone[0] != id, enumerate(bones)):
        children.extend(_get_children(cid, child, bones))
    return [(id, bone), *children]


def _remap_bone_ids(bones: list[tuple[int, Bone]]) -> dict[int, int]:
    return {old_id: new_id for new_id, (old_id, _) in enumerate(bones)}


def _make_sbc_cmd(cmd: SbcCmd, opt: SbcOpt) -> int:
    return int(cmd) | int(opt)


@dataclass(slots=True)
class Node:
    id: int
    parent: int
    world_mtx: np.ndarray
    mtx_slot: int | None = None
