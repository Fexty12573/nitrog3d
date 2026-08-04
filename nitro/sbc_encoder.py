from __future__ import annotations
from dataclasses import dataclass
from .model import ImportedSubModel, ImportedMesh, Bone
from .binary import BinaryWriter
from .sbc import SbcCmd, SbcOpt
from . import matrix as mat
import numpy as np
from collections import Counter, defaultdict


class SbcEncoder:
    def __init__(self, model: ImportedSubModel, mapping: BoneMapping):
        self.model = model
        self.bones = mapping.bones
        self.id_map = mapping.id_map
        self.id_map[-1] = 0
        self.nodes: list[Node] = []
        self.sbc = BinaryWriter()
        self.next_mtx_stack_id = 0
        self.last_desc: Node | None = None
        self.last_material: int | None = None
        self.current_bound_node = -1
        self.current_bound_mat = -1

        assert all(len(set(m.vertex_bone)) ==
                   1 for m in model.meshes), "multi matrix shapes not supported yet"
        mesh_nodes = [(m, m.vertex_bone[0]) for m in model.meshes]

        self.node_to_mesh: defaultdict[int, list[ImportedMesh]] \
            = defaultdict(list)
        for m, n in mesh_nodes:
            self.node_to_mesh[self.id_map[n]].append(m)

        self.mtx_read_nodes = {
            self.id_map[n] for m in model.meshes for n in set(m.vertex_bone)
        }

        # TODO: Use for "is STORE necessary" checks
        parents = Counter([b.parent for id, b in self.bones if b.parent != id])
        self.branches = {self.id_map[k] for k, v in parents.items() if v > 1}

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
            # All meshes bound to this node have the same visibility
            self._emit_node(bound_node, meshes[0].visible)

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
        self._emit_ret()

        return self.sbc.get_bytes()

    def _emit_nodedesc(self, node: Node):
        self.current_bound_node = node.id

        # local = mat.inverse(self.model.bones[bone.parent].world_mtx) @ bone.world_mtx

        opt = SbcOpt.NONE
        store = node.id in self.branches or node.id in self.mtx_read_nodes
        if store:
            opt |= SbcOpt.STORE

        # If the last node is NOT our parent, we need to restore the actual parent node's matrix slot
        restore = self.last_desc is not None and self.last_desc.id != node.parent
        if restore:
            opt |= SbcOpt.RESTORE

        cmd = _make_sbc_cmd(SbcCmd.NODEDESC, opt)

        self.sbc.write_u8(int(cmd))
        self.sbc.write_u8(node.id)
        self.sbc.write_u8(node.parent)
        self.sbc.write_u8(0)  # TODO: flags
        if store:
            self.sbc.write_u8(node.mtx_slot)
        if restore:
            self.sbc.write_u8(self.nodes[node.parent].mtx_slot)

    def _emit_node(self, node: int, visible: bool):
        cmd = _make_sbc_cmd(SbcCmd.NODE)
        self.sbc.write_u8(int(cmd))
        self.sbc.write_u8(node)
        self.sbc.write_u8(1 if visible else 0)

    def _emit_mtx(self, slot: int):
        cmd = _make_sbc_cmd(SbcCmd.MTX)
        self.sbc.write_u8(int(cmd))
        self.sbc.write_u8(slot)

    def _emit_posscale(self, inverse: bool = False):
        cmd = _make_sbc_cmd(
            SbcCmd.POSSCALE,
            SbcOpt.INVERSE if inverse else SbcOpt.NONE
        )
        self.sbc.write_u8(int(cmd))

    def _emit_mat(self, idx: int):
        cmd = _make_sbc_cmd(SbcCmd.MAT)
        self.sbc.write_u8(int(cmd))
        self.sbc.write_u8(idx)

    def _emit_shp(self, idx: int):
        cmd = _make_sbc_cmd(SbcCmd.SHP)
        self.sbc.write_u8(int(cmd))
        self.sbc.write_u8(idx)

    def _emit_ret(self):
        cmd = _make_sbc_cmd(SbcCmd.RET)
        self.sbc.write_u8(int(cmd))

    def _emit_nop(self):
        cmd = _make_sbc_cmd(SbcCmd.NOP)
        self.sbc.write_u8(int(cmd))


def _make_sbc_cmd(cmd: SbcCmd, opt: SbcOpt = SbcOpt.NONE) -> int:
    return int(cmd) | int(opt)


@dataclass(slots=True)
class Node:
    id: int
    parent: int
    world_mtx: np.ndarray
    mtx_slot: int | None = None


@dataclass(slots=True)
class BoneMapping:
    bones: list[tuple[int, Bone]]
    id_map: dict[int, int]

    @classmethod
    def create(cls, bones: list[Bone]):
        ordered = preorder_bones(bones)
        id_map = remap_bone_ids(ordered)
        return BoneMapping(ordered, id_map)
