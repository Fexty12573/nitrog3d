
import struct
from .sbc import SbcCmd, SbcOpt, NodeDescFlag


class SbcCommand:
    def __init__(self, kind: SbcCmd, opt: SbcOpt = SbcOpt.NONE):
        self.kind = kind
        self._opt = opt

    @property
    def opt(self) -> SbcOpt:
        return self._opt

    def to_bytes(self) -> bytes:
        return struct.pack('B', int(self.kind) | int(self.opt))

    def binds(self) -> int | None:
        return None

    # Whatever matrix needs to be "current" for this command
    def needs(self) -> int | None:
        return None

    def reads(self) -> list[int]:
        return []

    def set_restore(self, slot: int):
        raise NotImplementedError(f"{type(self).__name__} cannot restore")


class SbcNop(SbcCommand):
    def __init__(self):
        super().__init__(SbcCmd.NOP)


class SbcRet(SbcCommand):
    def __init__(self):
        super().__init__(SbcCmd.RET)


class SbcNode(SbcCommand):
    def __init__(self, node: int, visible: bool):
        super().__init__(SbcCmd.NODE)
        self.node = node
        self.visible = visible

    def to_bytes(self) -> bytes:
        return super().to_bytes() + struct.pack('BB', self.node, 1 if self.visible else 0)


class SbcMtx(SbcCommand):
    def __init__(self, node: int):
        super().__init__(SbcCmd.MTX)
        self.node = node
        self.slot: int | None = None

    def binds(self) -> int | None:
        return self.node

    def needs(self) -> int | None:
        return self.node

    def reads(self) -> list[int]:
        return [self.node]

    def set_restore(self, slot: int):
        self.slot = slot

    def to_bytes(self) -> bytes:
        return super().to_bytes() + struct.pack('B', self.slot)


class SbcMat(SbcCommand):
    def __init__(self, idx: int):
        super().__init__(SbcCmd.MAT)
        self.idx = idx

    def to_bytes(self) -> bytes:
        return super().to_bytes() + struct.pack('B', self.idx)


class SbcShp(SbcCommand):
    def __init__(self, idx: int):
        super().__init__(SbcCmd.SHP)
        self.idx = idx

    def to_bytes(self) -> bytes:
        return super().to_bytes() + struct.pack('B', self.idx)


class SbcStackCommand(SbcCommand):
    def __init__(self, kind: SbcCmd, node: int):
        super().__init__(kind)
        self.node = node
        self.store_slot: int | None = None
        self.restore_slot: int | None = None

    @property
    def opt(self) -> SbcOpt:
        opt = self._opt
        if self.store_slot is not None:
            opt |= SbcOpt.STORE
        if self.restore_slot is not None:
            opt |= SbcOpt.RESTORE
        return opt

    def store(self, slot: int):
        self.store_slot = slot

    def restore(self, slot: int):
        self.restore_slot = slot

    set_restore = restore

    def binds(self) -> int | None:
        return self.node


class SbcNodeDesc(SbcStackCommand):
    def __init__(self, node: int, parent: int, flags: NodeDescFlag = NodeDescFlag.NONE):
        super().__init__(SbcCmd.NODEDESC, node)
        self.parent = parent
        self.flags = flags

    @property
    def is_root(self) -> bool:
        return self.parent == self.node

    def needs(self) -> int | None:
        return None if self.is_root else self.parent

    def reads(self) -> list[int]:
        return [] if self.is_root else [self.parent]

    def ssc_parent(self):
        self.flags |= NodeDescFlag.SSC_PARENT

    def to_bytes(self) -> bytes:
        base = super().to_bytes() + struct.pack(
            'BBB',
            self.node,
            self.parent,
            int(self.flags)
        )
        if self.store_slot is not None:
            base += struct.pack('B', self.store_slot)
        if self.restore_slot is not None:
            base += struct.pack('B', self.restore_slot)
        return base


class SbcBb(SbcStackCommand):
    def __init__(self, node: int):
        super().__init__(SbcCmd.BB, node)

    def to_bytes(self) -> bytes:
        base = super().to_bytes() + struct.pack('B', self.node)
        if self.store_slot is not None:
            base += struct.pack('B', self.store_slot)
        if self.restore_slot is not None:
            base += struct.pack('B', self.restore_slot)
        return base


class SbcBbY(SbcStackCommand):
    def __init__(self, node: int):
        super().__init__(SbcCmd.BBY, node)

    def to_bytes(self) -> bytes:
        base = super().to_bytes() + struct.pack('B', self.node)
        if self.store_slot is not None:
            base += struct.pack('B', self.store_slot)
        if self.restore_slot is not None:
            base += struct.pack('B', self.restore_slot)
        return base


class SbcNodeMix(SbcCommand):
    def __init__(self, store: int, ops: list[tuple[int, int, int]]):
        super().__init__(SbcCmd.NODEMIX)
        self.store_slot = store
        self.ops = ops

    def reads(self) -> list[int]:
        return [node for _, node, _ in self.ops]

    def to_bytes(self) -> bytes:
        base = super().to_bytes() + struct.pack('BB', self.store_slot, len(self.ops))
        for src, node, ratio in self.ops:
            base += struct.pack('BBB', src, node, ratio)
        return base


class SbcCallDl(SbcCommand):
    def __init__(self, offset: int, size: int):
        super().__init__(SbcCmd.CALLDL)
        self.offset = offset
        self.size = size

    def to_bytes(self) -> bytes:
        return super().to_bytes() + struct.pack('<II', self.offset, self.size)


class SbcPosScale(SbcCommand):
    def __init__(self, inverse: bool = False):
        super().__init__(SbcCmd.POSSCALE, SbcOpt.INVERSE if inverse else SbcOpt.NONE)


class SbcEnvMap(SbcCommand):
    def __init__(self, mat: int):
        super().__init__(SbcCmd.ENVMAP)
        self.mat = mat

    def to_bytes(self) -> bytes:
        return super().to_bytes() + struct.pack('BB', self.mat, 0)


class SbcPrjMap(SbcCommand):
    def __init__(self, mat: int):
        super().__init__(SbcCmd.PRJMAP)
        self.mat = mat

    def to_bytes(self) -> bytes:
        return super().to_bytes() + struct.pack('BB', self.mat, 0)
