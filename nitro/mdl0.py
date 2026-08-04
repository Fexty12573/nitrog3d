
from __future__ import annotations
from .binary import BinaryReader, BinaryWriter, FX16_SCALE, float_to_bgr555
from .dictionary import read_dictionary, write_dictionary, make_dictionary
from .tex0 import TexImageParam, TexFmt
from enum import IntEnum


def _hasflag(field: int, flag: int) -> bool:
    return (field & flag) != 0


class SrtFlag(IntEnum):
    TRANSLATION_ZERO = 1 << 0
    ROTATION_ZERO = 1 << 1
    SCALE_ONE = 1 << 2
    HAS_PIVOT = 1 << 3
    PIVOT_IDX_MASK = 0xF << 4
    PIVOT_NEGATIVE = 1 << 8
    SIGN_REVC = 1 << 9
    SIGN_REVD = 1 << 10


class MatFlag(IntEnum):
    NONE = 0
    TEXMTX_USE = 1 << 0
    TEXMTX_SCALE_ONE = 1 << 1
    TEXMTX_ROTATION_ZERO = 1 << 2
    TEXMTX_TRANSLATION_ZERO = 1 << 3
    ORIGINAL_WH_SAME = 1 << 4
    WIREFRAME = 1 << 5
    DIFFUSE = 1 << 6
    AMBIENT = 1 << 7
    VTXCOLOR = 1 << 8
    SPECULAR = 1 << 9
    EMISSION = 1 << 10
    SHININESS = 1 << 11
    TEXPLTTBASE = 1 << 12
    EFFECTMTX = 1 << 13


class CullMode(IntEnum):
    ALL = 0
    FRONT = 1
    BACK = 2
    NONE = 3


class PolygonMode(IntEnum):
    MODULATE = 0
    DECAL = 1
    TOON = 2
    SHADOW = 3


class PolygonAttr:
    def __init__(self, v: int):
        self.v = v

    @property
    def lights(self) -> int:
        return self.v & 0xF

    @property
    def mode(self) -> PolygonMode:
        return PolygonMode((self.v >> 4) & 0x3)

    @property
    def cull_mode(self) -> CullMode:
        return CullMode((self.v >> 6) & 0x3)

    @property
    def alpha(self) -> int:
        return (self.v >> 16) & 0x1f

    @property
    def id(self) -> int:
        return (self.v >> 24) & 0x3F

    @property
    def xlu_depth_update(self) -> bool:
        return ((self.v >> 11) & 1) != 0

    @property
    def far_clipping(self) -> bool:
        return ((self.v >> 12) & 1) != 0

    @property
    def disp_1dot(self) -> bool:
        return ((self.v >> 13) & 1) != 0

    @property
    def depthtest_decal(self) -> bool:
        return ((self.v >> 14) & 1) != 0

    @property
    def fog(self) -> bool:
        return ((self.v >> 15) & 1) != 0

    def __eq__(self, value):
        if isinstance(value, PolygonAttr):
            return self.v == value.v
        if isinstance(value, int):
            return self.v == value
        return False

    @classmethod
    def build(cls,
              mode: PolygonMode,
              cull: CullMode,
              alpha: int,
              lights: int = 0,
              id: int = 0,
              xlu_depth_update: bool = False,
              far_clipping: bool = False,
              disp_1dot: bool = False,
              depthtest_decal: bool = False,
              fog: bool = False) -> PolygonAttr:
        v = 0
        v |= (lights & 0xF)
        v |= (int(mode) & 0x3) << 4
        v |= (int(cull) & 0x3) << 6
        v |= (1 if xlu_depth_update else 0) << 11
        v |= (1 if far_clipping else 0) << 12
        v |= (1 if disp_1dot else 0) << 13
        v |= (1 if depthtest_decal else 0) << 14
        v |= (1 if fog else 0) << 15
        v |= (alpha & 0x1F) << 16
        v |= (id & 0x3F) << 24
        return cls(v)


class MDL0:
    SIGNATURE = "MDL0"

    def __init__(self, r: BinaryReader):
        base = r.tell()
        sig = r.read_str(4)
        if sig != self.SIGNATURE:
            raise ValueError(f"Expected MDL0 signature, got {sig}")

        self.section_size = r.read_u32()
        self.dict = read_dictionary(r, lambda rd: rd.read_u32())
        self.models: list[Model] = []
        for offset in self.dict.data:
            r.seek(base + offset)
            self.models.append(Model(r))

    def write(self, w: BinaryWriter):
        base = w.tell()
        w.write_str(self.SIGNATURE)

        pos_size = w.tell()
        w.write_u32(0)
        write_dictionary(w, self.dict, lambda wr, v: wr.write_u32(v))
        for i, model in enumerate(self.models):
            self.dict.values()[i] = w.tell() - base
            model.write(w)
        end = w.tell()
        w.patch_u32(pos_size, end - base)

        # Rewrite dictionary with updated offsets
        w.seek(base + 8)
        write_dictionary(w, self.dict, lambda wr, v: wr.write_u32(v))
        w.seek(end)

    @classmethod
    def build(cls, models: dict[str, Model]) -> MDL0:
        m = cls.__new__(cls)
        m.dict = make_dictionary({n: 0 for n in models.keys()}, 4)
        m.models = list(models.values())
        m.section_size = 0
        return m

    def __iter__(self) -> zip[tuple[str, Model]]:
        return zip(self.dict.keys(), self.models)


class Model:
    def __init__(self, r: BinaryReader):
        base = r.tell()
        self.size = r.read_u32()
        self.sbc_offset = r.read_u32()
        self.mat_offset = r.read_u32()
        self.shp_offset = r.read_u32()
        self.evp_offset = r.read_u32()
        self.info = ModelInfo(r)
        self.nodes = NodeSet(r)

        r.seek(base + self.sbc_offset)
        self.sbc = r.read_bytes(self.mat_offset - self.sbc_offset)

        r.seek(base + self.mat_offset)
        self.materials = MaterialSet(r)

        r.seek(base + self.shp_offset)
        self.shapes = ShapeSet(r)

        self.evp_matrices = None
        if self.evp_offset != self.size and self.evp_offset > 0:
            r.seek(base + self.evp_offset)
            self.evp_matrices = EvpMatrices(r, len(self.nodes))

    def write(self, w: BinaryWriter):
        base = w.tell()
        w.write_u32s([0, 0, 0, 0, 0])  # size and offsets
        self.info.write(w)
        self.nodes.write(w)

        w.patch_u32(base + 4, w.tell() - base)  # Sbc offset
        w.write_bytes(self.sbc)

        w.patch_u32(base + 8, w.tell() - base)  # MaterialSet offset
        self.materials.write(w)

        w.patch_u32(base + 12, w.tell() - base)  # ShapeSet offset
        self.shapes.write(w)

        w.patch_u32(base + 16, w.tell() - base)  # EvpMatrices offset
        if self.evp_matrices is not None:
            self.evp_matrices.write(w)

        w.patch_u32(base, w.tell() - base)  # Size

    @classmethod
    def build(cls, info: ModelInfo,
              nodes: NodeSet,
              sbc: bytes,
              materials: MaterialSet,
              shapes: ShapeSet,
              evp_matrices: EvpMatrices | None = None) -> Model:
        model = cls.__new__(cls)
        model.info = info
        model.nodes = nodes
        model.sbc = sbc
        model.materials = materials
        model.shapes = shapes
        model.evp_matrices = evp_matrices
        return model


class ModelInfo:
    def __init__(self, r: BinaryReader):
        self.sbc_type = r.read_u8()
        self.scaling_rule = r.read_u8()
        self.tex_mtx_mode = r.read_u8()
        self.node_count = r.read_u8()
        self.mat_count = r.read_u8()
        self.shape_count = r.read_u8()
        self.first_unused_mtx_stack_id = r.read_u8()
        r.skip(1)  # Padding?
        self.pos_scale = r.read_fx32()
        self.inv_pos_scale = r.read_fx32()
        self.vertex_count = r.read_u16()
        self.polygon_count = r.read_u16()
        self.triangle_count = r.read_u16()
        self.quad_count = r.read_u16()
        self.box_x = r.read_fx16()
        self.box_y = r.read_fx16()
        self.box_z = r.read_fx16()
        self.box_w = r.read_fx16()
        self.box_h = r.read_fx16()
        self.box_d = r.read_fx16()
        self.box_pos_scale = r.read_fx32()
        self.box_inv_pos_scale = r.read_fx32()

    def write(self, w: BinaryWriter):
        w.write_u8(self.sbc_type)
        w.write_u8(self.scaling_rule)
        w.write_u8(self.tex_mtx_mode)
        w.write_u8(self.node_count)
        w.write_u8(self.mat_count)
        w.write_u8(self.shape_count)
        w.write_u8(self.first_unused_mtx_stack_id)
        w.write_u8(0)
        w.write_fx32(self.pos_scale)
        w.write_fx32(self.inv_pos_scale)
        w.write_u16(self.vertex_count)
        w.write_u16(self.polygon_count)
        w.write_u16(self.triangle_count)
        w.write_u16(self.quad_count)
        w.write_fx16(self.box_x)
        w.write_fx16(self.box_y)
        w.write_fx16(self.box_z)
        w.write_fx16(self.box_w)
        w.write_fx16(self.box_h)
        w.write_fx16(self.box_d)
        w.write_fx32(self.box_pos_scale)
        w.write_fx32(self.box_inv_pos_scale)

    @classmethod
    def builder(cls) -> _ModelInfoBuilder:
        return _ModelInfoBuilder()


class _ModelInfoBuilder:
    def __init__(self):
        self._node_count = 0
        self._mat_count = 0
        self._shape_count = 0
        self._first_unused_mtx_stack_id = 0
        self._pos_scale = 1.0
        self._inv_pos_scale = 1.0
        self._vertex_count = 0
        self._polygon_count = 0
        self._triangle_count = 0
        self._quad_count = 0
        self._box_x = 0.0
        self._box_y = 0.0
        self._box_z = 0.0
        self._box_w = 1.0
        self._box_h = 1.0
        self._box_d = 1.0
        self._box_pos_scale = 1.0
        self._box_inv_pos_scale = 1.0

    def node_count(self, count: int) -> _ModelInfoBuilder:
        self._node_count = count
        return self

    def mat_count(self, count: int) -> _ModelInfoBuilder:
        self._mat_count = count
        return self

    def shape_count(self, count: int) -> _ModelInfoBuilder:
        self._shape_count = count
        return self

    def first_unused_mtx_stack_id(self, idx: int) -> _ModelInfoBuilder:
        self._first_unused_mtx_stack_id = idx
        return self

    def pos_scale(self, scale: float) -> _ModelInfoBuilder:
        self._pos_scale = scale
        self._inv_pos_scale = 1.0 / scale
        return self

    def vertex_count(self, count: int) -> _ModelInfoBuilder:
        self._vertex_count = count
        return self

    def polygon_count(self, count: int) -> _ModelInfoBuilder:
        self._polygon_count = count
        return self

    def triangle_count(self, count: int) -> _ModelInfoBuilder:
        self._triangle_count = count
        return self

    def quad_count(self, count: int) -> _ModelInfoBuilder:
        self._quad_count = count
        return self

    def bounding_box(self, x: float, y: float, z: float, w: float, h: float, d: float) -> _ModelInfoBuilder:
        self._box_x = x
        self._box_y = y
        self._box_z = z
        self._box_w = w
        self._box_h = h
        self._box_d = d
        return self

    def box_pos_scale(self, scale: float) -> _ModelInfoBuilder:
        self._box_pos_scale = scale
        self._box_inv_pos_scale = 1.0 / scale
        return self

    def build(self) -> ModelInfo:
        info = ModelInfo.__new__(ModelInfo)
        info.sbc_type = 0  # NNS_G3D_SBCTYPE_NORMAL
        info.scaling_rule = 0  # NNS_G3D_SCALINGRULE_STANDARD TODO: Maybe make this a parameter
        info.tex_mtx_mode = 0  # NNS_G3D_TEXMTXMODE_MAYA
        info.node_count = self._node_count
        info.mat_count = self._mat_count
        info.shape_count = self._shape_count
        info.first_unused_mtx_stack_id = self._first_unused_mtx_stack_id
        info.pos_scale = self._pos_scale
        info.inv_pos_scale = self._inv_pos_scale
        info.vertex_count = self._vertex_count
        info.polygon_count = self._polygon_count
        info.triangle_count = self._triangle_count
        info.quad_count = self._quad_count
        info.box_x = self._box_x
        info.box_y = self._box_y
        info.box_z = self._box_z
        info.box_w = self._box_w
        info.box_h = self._box_h
        info.box_d = self._box_d
        info.box_pos_scale = self._box_pos_scale
        info.box_inv_pos_scale = self._box_inv_pos_scale
        return info


class NodeData:
    """Static SRT data of a joint"""

    def __init__(self, r: BinaryReader):
        self.flag = r.read_u16()
        self._00 = r.read_s16()

        # Translation
        self.tx = self.ty = self.tz = 0
        if not self.hasflag(SrtFlag.TRANSLATION_ZERO):
            self.tx = r.read_fx32()
            self.ty = r.read_fx32()
            self.tz = r.read_fx32()

        self._01 = self._02 = self._10 = self._11 = self._12 = self._20 = self._21 = self._22 = 0.0

        # Full 3x3 rotation
        if not self.hasflag(SrtFlag.ROTATION_ZERO | SrtFlag.HAS_PIVOT):
            self._01 = r.read_fx16()
            self._02 = r.read_fx16()
            self._10 = r.read_fx16()
            self._11 = r.read_fx16()
            self._12 = r.read_fx16()
            self._20 = r.read_fx16()
            self._21 = r.read_fx16()
            self._22 = r.read_fx16()

        # Pivot rotation
        self.a = self.b = 0.0
        if self.hasflag(SrtFlag.HAS_PIVOT) and not self.hasflag(SrtFlag.ROTATION_ZERO):
            self.a = r.read_fx16()
            self.b = r.read_fx16()

        # Scale
        self.sx = self.sy = self.sz = 0.0
        self.inv_sx = self.inv_sy = self.inv_sz = 0.0
        if not self.hasflag(SrtFlag.SCALE_ONE):
            self.sx = r.read_fx32()
            self.sy = r.read_fx32()
            self.sz = r.read_fx32()
            self.inv_sx = r.read_fx32()
            self.inv_sy = r.read_fx32()
            self.inv_sz = r.read_fx32()

    def write(self, w: BinaryWriter):
        w.write_u16(self.flag)
        w.write_u16(self._00)
        if not self.hasflag(SrtFlag.TRANSLATION_ZERO):
            w.write_fx32(self.tx)
            w.write_fx32(self.ty)
            w.write_fx32(self.tz)
        if not self.hasflag(SrtFlag.ROTATION_ZERO | SrtFlag.HAS_PIVOT):
            w.write_fx16(self._01)
            w.write_fx16(self._02)
            w.write_fx16(self._10)
            w.write_fx16(self._11)
            w.write_fx16(self._12)
            w.write_fx16(self._20)
            w.write_fx16(self._21)
            w.write_fx16(self._22)
        if self.hasflag(SrtFlag.HAS_PIVOT) and not self.hasflag(SrtFlag.ROTATION_ZERO):
            w.write_fx16(self.a)
            w.write_fx16(self.b)
        if not self.hasflag(SrtFlag.SCALE_ONE):
            w.write_fx32(self.sx)
            w.write_fx32(self.sy)
            w.write_fx32(self.sz)
            w.write_fx32(self.inv_sx)
            w.write_fx32(self.inv_sy)
            w.write_fx32(self.inv_sz)

    @classmethod
    def builder(cls) -> _NodeDataBuilder:
        return _NodeDataBuilder()

    def hasflag(self, flag: SrtFlag) -> bool:
        return _hasflag(self.flag, flag)

    def pivot_idx(self) -> tuple[int, int]:
        idx = (self.flag & SrtFlag.PIVOT_IDX_MASK) >> 4
        return (idx // 3, idx % 3)

    def translation(self) -> tuple[float, float, float]:
        return (self.tx, self.ty, self.tz)

    def rot_mtx(self) -> list[float]:
        return [
            self._00 / FX16_SCALE, self._01, self._02,
            self._10, self._11, self._12,
            self._20, self._21, self._22
        ]


class _NodeDataBuilder:
    def __init__(self):
        self._flag = SrtFlag.SCALE_ONE | SrtFlag.ROTATION_ZERO | SrtFlag.TRANSLATION_ZERO
        self.t = (0.0, 0.0, 0.0)
        self.rot = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        self.a = self.b = 0.0
        self.s = (1.0, 1.0, 1.0)

    def translate(self, translation: tuple[float, float, float]) -> _NodeDataBuilder:
        self.t = translation
        if translation != (0.0, 0.0, 0.0):
            self._flag &= ~SrtFlag.TRANSLATION_ZERO
        return self

    def rotate(self, rot3x3: list[float]) -> _NodeDataBuilder:
        self.rot = rot3x3
        self._flag &= ~(SrtFlag.ROTATION_ZERO | SrtFlag.HAS_PIVOT)
        return self

    def pivot(self, idx: int, a: float, b: float) -> _NodeDataBuilder:
        self.a = a
        self.b = b
        self._flag = (self._flag & 0xFF0F) | ((idx & 0xF) << 4)
        self._flag |= SrtFlag.HAS_PIVOT
        self._flag &= ~SrtFlag.ROTATION_ZERO
        return self

    def scale(self, scale: tuple[float, float, float]) -> _NodeDataBuilder:
        self.s = scale
        if scale != (1.0, 1.0, 1.0):
            self._flag &= ~SrtFlag.SCALE_ONE
        return self

    def build(self) -> NodeData:
        n = NodeData.__new__(NodeData)
        n.flag = int(self._flag)
        n._00 = int(self.rot[0] * FX16_SCALE)
        n.tx, n.ty, n.tz = self.t
        n.sx, n.sy, n.sz = self.s
        n.inv_sx, n.inv_sy, n.inv_sz = 1 / n.sx, 1 / n.sy, 1 / n.sz
        n._01 = self.rot[1]
        n._02 = self.rot[2]
        n._10 = self.rot[3]
        n._11 = self.rot[4]
        n._12 = self.rot[5]
        n._20 = self.rot[6]
        n._21 = self.rot[7]
        n._22 = self.rot[8]
        n.a = self.a
        n.b = self.b
        return n


class NodeSet:
    def __init__(self, r: BinaryReader):
        base = r.tell()
        self.dict = read_dictionary(r, lambda rd: rd.read_u32())
        resume = r.tell()
        self.nodes: list[NodeData] = []
        for offset in self.dict.data:
            r.seek(base + offset)
            self.nodes.append(NodeData(r))
        r.seek(resume)

    def write(self, w: BinaryWriter):
        base = w.tell()
        write_dictionary(w, self.dict, lambda wr, v: wr.write_u32(v))
        for i, node in enumerate(self.nodes):
            self.dict.values()[i] = w.tell() - base
            node.write(w)
        end = w.tell()
        w.seek(base)
        write_dictionary(w, self.dict, lambda wr, v: wr.write_u32(v))
        w.seek(end)

    @classmethod
    def build(cls, nodes: dict[str, NodeData]) -> NodeSet:
        ns = cls.__new__(cls)
        ns.nodes = list(nodes.values())
        ns.dict = make_dictionary({nn: 0 for nn in nodes.keys()}, 4)
        return ns

    def __len__(self) -> int:
        return len(self.nodes)


class Material:
    def __init__(self, r: BinaryReader):
        self.tag = r.read_u16()
        self.size = r.read_u16()
        self.diff = r.read_u16()
        self.amb = r.read_u16()
        self.spec = r.read_u16()
        self.emi = r.read_u16()
        self.poly_attr = PolygonAttr(r.read_u32())
        self.poly_attr_mask = r.read_u32()
        self.tex_image_param = TexImageParam(r.read_u32())
        self.tex_image_param_mask = r.read_u32()
        self.tex_pltt_base = r.read_u16()
        self.flag = r.read_u16()
        self.orig_width = r.read_u16()
        self.orig_height = r.read_u16()
        self.mag_w = r.read_fx32()
        self.mag_h = r.read_fx32()
        self.scale_s = self.scale_t = 1.0
        self.rot_sin = 0.0
        self.rot_cos = 1.0
        self.trans_s = self.trans_t = 0.0
        self.effect_mtx = None
        if not self.hasflag(MatFlag.TEXMTX_SCALE_ONE):
            self.scale_s = r.read_fx32()
            self.scale_t = r.read_fx32()
        if not self.hasflag(MatFlag.TEXMTX_ROTATION_ZERO):
            self.rot_sin = r.read_fx16()
            self.rot_cos = r.read_fx16()
        if not self.hasflag(MatFlag.TEXMTX_TRANSLATION_ZERO):
            self.trans_s = r.read_fx32()
            self.trans_t = r.read_fx32()
        if self.hasflag(MatFlag.EFFECTMTX):
            self.effect_mtx = r.read_fx32s(16)

    def write(self, w: BinaryWriter):
        w.write_u16(self.tag)
        w.write_u16(self.size)
        w.write_u16(self.diff)
        w.write_u16(self.amb)
        w.write_u16(self.spec)
        w.write_u16(self.emi)
        w.write_u32(self.poly_attr.v)
        w.write_u32(self.poly_attr_mask)
        w.write_u32(self.tex_image_param.v)
        w.write_u32(self.tex_image_param_mask)
        w.write_u16(self.tex_pltt_base)
        w.write_u16(self.flag)
        w.write_u16(self.orig_width)
        w.write_u16(self.orig_height)
        w.write_fx32(self.mag_w)
        w.write_fx32(self.mag_h)
        if not self.hasflag(MatFlag.TEXMTX_SCALE_ONE):
            w.write_fx32(self.scale_s)
            w.write_fx32(self.scale_t)
        if not self.hasflag(MatFlag.TEXMTX_ROTATION_ZERO):
            w.write_fx16(self.rot_sin)
            w.write_fx16(self.rot_cos)
        if not self.hasflag(MatFlag.TEXMTX_TRANSLATION_ZERO):
            w.write_fx32(self.trans_s)
            w.write_fx32(self.trans_t)
        if self.hasflag(MatFlag.EFFECTMTX):
            w.write_fx32s(self.effect_mtx)

    @classmethod
    def builder(cls) -> _MatBuilder:
        return _MatBuilder()

    def hasflag(self, flag: MatFlag) -> bool:
        return _hasflag(self.flag, flag)

    @property
    def tex_width(self) -> int:
        return self.tex_image_param.width

    @property
    def tex_height(self) -> int:
        return self.tex_image_param.height

    @property
    def tex_format(self) -> TexFmt:
        return self.tex_image_param.format

    @property
    def cull_mode(self) -> CullMode:
        return self.poly_attr.cull_mode

    @property
    def alpha(self) -> int:
        return self.poly_attr.alpha


class _MatBuilder:
    def __init__(self):
        self._tag = 0
        self._diff = (0.0, 0.0, 0.0)
        self._amb = (0.0, 0.0, 0.0)
        self._spec = (0.0, 0.0, 0.0)
        self._emi = (0.0, 0.0, 0.0)
        self._poly_attr = PolygonAttr(0)
        self._poly_attr_mask = 0
        self._tex_image_param = TexImageParam(0)
        self._tex_image_param_mask = 0
        self._tex_pltt_base = 0
        self._flag = MatFlag.TEXMTX_SCALE_ONE | MatFlag.TEXMTX_ROTATION_ZERO | MatFlag.TEXMTX_TRANSLATION_ZERO
        self._orig_width = 0
        self._orig_height = 0
        self._mag_w = 1.0
        self._mag_h = 1.0
        self._scale_s = 1.0
        self._scale_t = 1.0
        self._rot_sin = 0.0
        self._rot_cos = 1.0
        self._trans_s = 0.0
        self._trans_t = 0.0
        self._effect_mtx: list[float] | None = None

    def tag(self, tag: int) -> _MatBuilder:
        self._tag = tag
        return self

    def diffuse(self, diffuse: tuple[float, float, float]) -> _MatBuilder:
        self._diff = diffuse
        return self

    def ambient(self, ambient: tuple[float, float, float]) -> _MatBuilder:
        self._amb = ambient
        return self

    def specular(self, specular: tuple[float, float, float]) -> _MatBuilder:
        self._spec = specular
        return self

    def emissive(self, emissive: tuple[float, float, float]) -> _MatBuilder:
        self._emi = emissive
        return self

    def poly_attr(self, attr: PolygonAttr, mask: int) -> _MatBuilder:
        self._poly_attr = attr
        self._poly_attr_mask = mask
        return self

    def tex_image_param(self, param: TexImageParam, mask: int) -> _MatBuilder:
        self._tex_image_param = param
        self._tex_image_param_mask = mask
        return self

    def tex_pltt_base(self, base: int) -> _MatBuilder:
        self._tex_pltt_base = base
        return self

    def orig_size(self, width: int, height: int) -> _MatBuilder:
        self._orig_width = width
        self._orig_height = height
        return self

    def mag_size(self, width: float, height: float) -> _MatBuilder:
        self._mag_w = width
        self._mag_h = height
        return self

    def tex_scale(self, scale_s: float, scale_t: float) -> _MatBuilder:
        self._scale_s = scale_s
        self._scale_t = scale_t
        if scale_s != 1.0 or scale_t != 1.0:
            self._flag &= ~MatFlag.TEXMTX_SCALE_ONE
        return self

    def tex_rotation(self, sin: float, cos: float) -> _MatBuilder:
        self._rot_sin = sin
        self._rot_cos = cos
        if sin != 0.0 or cos != 1.0:
            self._flag &= ~MatFlag.TEXMTX_ROTATION_ZERO
        return self

    def tex_translation(self, trans_s: float, trans_t: float) -> _MatBuilder:
        self._trans_s = trans_s
        self._trans_t = trans_t
        if trans_s != 0.0 or trans_t != 0.0:
            self._flag &= ~MatFlag.TEXMTX_TRANSLATION_ZERO
        return self

    def effect_matrix(self, mtx: list[float] | None) -> _MatBuilder:
        self._effect_mtx = mtx
        if mtx is not None:
            self._flag |= MatFlag.EFFECTMTX
        return self

    def build(self) -> Material:
        mat = Material.__new__(Material)
        mat.tag = self._tag
        mat.diff = float_to_bgr555(*self._diff)
        mat.amb = float_to_bgr555(*self._amb)
        mat.spec = float_to_bgr555(*self._spec)
        mat.emi = float_to_bgr555(*self._emi)
        mat.poly_attr = self._poly_attr
        mat.poly_attr_mask = self._poly_attr_mask
        mat.tex_image_param = self._tex_image_param
        mat.tex_image_param_mask = self._tex_image_param_mask
        mat.tex_pltt_base = self._tex_pltt_base
        mat.flag = int(self._flag)
        mat.orig_width = self._orig_width
        mat.orig_height = self._orig_height
        mat.mag_w = self._mag_w
        mat.mag_h = self._mag_h
        mat.scale_s = self._scale_s
        mat.scale_t = self._scale_t
        mat.rot_sin = self._rot_sin
        mat.rot_cos = self._rot_cos
        mat.trans_s = self._trans_s
        mat.trans_t = self._trans_t
        mat.effect_mtx = self._effect_mtx

        size = 44
        if not mat.hasflag(MatFlag.TEXMTX_SCALE_ONE):
            size += 8
        if not mat.hasflag(MatFlag.TEXMTX_ROTATION_ZERO):
            size += 4
        if not mat.hasflag(MatFlag.TEXMTX_TRANSLATION_ZERO):
            size += 8
        if mat.hasflag(MatFlag.EFFECTMTX):
            size += 64

        mat.size = size

        return mat


class MaterialSet:
    def __init__(self, r: BinaryReader):
        base = r.tell()
        r.skip(4)  # offsets to tex2mat and pltt2mat dicts

        self.dict = read_dictionary(r, lambda rd: rd.read_u32())
        resume = r.tell()
        self.materials: list[Material] = []
        for offset in self.dict.data:
            r.seek(base + offset)
            self.materials.append(Material(r))
        r.seek(resume)

        self.dict_tex_to_mat = read_dictionary(r, lambda rd: TexToMatData(rd))
        for entry in self.dict_tex_to_mat.data:
            start = r.tell()
            r.seek(base + entry.offset)
            entry.materials = list(r.read_bytes(entry.mat_count))
            r.seek(start)

        self.dict_pltt_to_mat = read_dictionary(r, lambda rd: TexToMatData(rd))
        for entry in self.dict_pltt_to_mat.data:
            start = r.tell()
            r.seek(base + entry.offset)
            entry.materials = list(r.read_bytes(entry.mat_count))
            r.seek(start)

    def write(self, w: BinaryWriter):
        base = w.tell()
        pos_tex_offset = w.tell()
        w.write_u16(0)
        pos_pltt_offset = w.tell()
        w.write_u16(0)

        write_dictionary(w, self.dict, lambda wr, v: wr.write_u32(v))
        w.patch_u16(pos_tex_offset, w.tell() - base)
        write_dictionary(w, self.dict_tex_to_mat, lambda wr, v: v.write(wr))
        w.patch_u16(pos_pltt_offset, w.tell() - base)
        write_dictionary(w, self.dict_pltt_to_mat, lambda wr, v: v.write(wr))

        # Write Material lists and update entry offsets
        for entry in self.dict_tex_to_mat.values():
            entry.offset = w.tell() - base
            w.write_bytes(bytes(entry.materials))
        for entry in self.dict_pltt_to_mat.values():
            entry.offset = w.tell() - base
            w.write_bytes(bytes(entry.materials))
        w.align(4)

        # Update material offsets
        for i, mat in enumerate(self.materials):
            self.dict.data[i] = w.tell() - base
            mat.write(w)

        end = w.tell()

        # Re-write the 3 dictionaries with their updated entries
        w.seek(base + 4)
        write_dictionary(w, self.dict, lambda wr, v: wr.write_u32(v))
        write_dictionary(w, self.dict_tex_to_mat, lambda wr, v: v.write(wr))
        write_dictionary(w, self.dict_pltt_to_mat, lambda wr, v: v.write(wr))
        w.seek(end)

    def texture_name(self, mat: int) -> str | None:
        for name, entry in self.dict_tex_to_mat:
            if mat in entry.materials:
                return name
        return None

    def palette_name(self, mat: int) -> str | None:
        for name, entry in self.dict_pltt_to_mat:
            if mat in entry.materials:
                return name
        return None

    def __iter__(self):
        return self.materials.__iter__()

    def __len__(self) -> int:
        return len(self.materials)

    @classmethod
    def build(cls, names: list[str], materials: list[Material],
              tex_names: list[str | None] | None = None,
              pltt_names: list[str | None] | None = None) -> MaterialSet:
        ms = cls.__new__(cls)
        ms.materials = list(materials)
        ms.dict = make_dictionary({n: 0 for n in names}, 4)

        def invert(assoc: list[str | None] | None):
            groups: dict[str, list[int]] = {}
            for i, n in enumerate(assoc or []):
                if n is not None:
                    groups.setdefault(n, []).append(i)
            keys = sorted(groups)  # tex/pltt dicts are name-sorted
            return make_dictionary({k: TexToMatData.build(groups[k]) for k in keys}, 4)

        ms.dict_tex_to_mat = invert(tex_names)
        ms.dict_pltt_to_mat = invert(pltt_names)
        return ms


class TexToMatData:
    """Maps a texture (or palette) name to the material indices that use it"""

    def __init__(self, r: BinaryReader):
        self.offset = r.read_u16()
        self.mat_count = r.read_u8() & 0x7F
        self.flags = r.read_u8()
        self.materials: list[int] = []  # Filled in by MaterialSet

    def write(self, w: BinaryWriter):
        w.write_u16(self.offset)
        w.write_u8(len(self.materials))
        w.write_u8(self.flags)

    @classmethod
    def build(cls, materials: list[int], flags: int = 0) -> TexToMatData:
        v = cls.__new__(cls)
        v.offset = 0
        v.mat_count = len(materials)
        v.flags = flags
        v.materials = materials
        return v


class Shape:
    def __init__(self, r: BinaryReader):
        base = r.tell()
        self.tag = r.read_u16()
        self.size = r.read_u16()
        self.flag = r.read_u32()
        self.dl_offset = r.read_u32()
        self.dl_size = r.read_u32()
        resume = r.tell()
        r.seek(base + self.dl_offset)
        self.dl = r.read_bytes(self.dl_size)
        r.seek(resume)

    def write(self, w: BinaryWriter):
        w.write_u16(self.tag)
        w.write_u16(self.size)
        w.write_u32(self.flag)
        w.write_u32(self.dl_offset)
        w.write_u32(self.dl_size)

    @classmethod
    def build(cls, tag: int, flag: int, dl: bytes) -> Shape:
        s = cls.__new__(cls)
        s.tag = tag
        s.size = 16
        s.flag = flag
        s.dl = dl
        s.dl_offset = 0
        s.dl_size = len(dl)
        return s


class ShapeSet:
    def __init__(self, r: BinaryReader):
        base = r.tell()
        self.dict = read_dictionary(r, lambda rd: rd.read_u32())
        resume = r.tell()
        self.shapes: list[Shape] = []
        for offset in self.dict.values():
            r.seek(base + offset)
            self.shapes.append(Shape(r))
        r.seek(resume)

    def write(self, w: BinaryWriter):
        base = w.tell()
        write_dictionary(w, self.dict, lambda wr, v: wr.write_u32(v))
        for i, shape in enumerate(self.shapes):
            self.dict.data[i] = w.tell() - base
            shape.write(w)

        # Display-list bytes follow the shape descriptors
        for i, shape in enumerate(self.shapes):
            shape.dl_offset = w.tell() - base - self.dict.data[i]
            shape.dl_size = len(shape.dl)
            w.write_bytes(shape.dl)

        end = w.tell()

        # Re-write both dictionary and shapes with updated offsets/sizes
        w.seek(base)
        write_dictionary(w, self.dict, lambda wr, v: wr.write_u32(v))
        for shape in self.shapes:
            shape.write(w)
        w.seek(end)

    def __len__(self) -> int:
        return len(self.shapes)

    @classmethod
    def build(cls, shapes: dict[str, Shape]) -> ShapeSet:
        s = cls.__new__(cls)
        s.shapes = list(shapes.values())
        s.dict = make_dictionary({n: 0 for n in shapes.keys()}, 4)
        return s


class Envelope:
    def __init__(self, r: BinaryReader):
        self.inv_m = r.read_fx32s(12)  # Row major 4x3 matrix
        self.inv_n = r.read_fx32s(9)  # Row major 3x3 matrix

    def write(self, w: BinaryWriter):
        w.write_fx32s(self.inv_m)
        w.write_fx32s(self.inv_n)

    @classmethod
    def build(cls, inv_m: list[float], inv_n: list[float]) -> Envelope:
        env = cls.__new__(cls)
        env.inv_m = inv_m
        env.inv_n = inv_n
        return env


class EvpMatrices:
    def __init__(self, r: BinaryReader, node_count: int):
        self.m = [Envelope(r) for _ in range(node_count)]

    def write(self, w: BinaryWriter):
        for evp in self.m:
            evp.write(w)

    def __len__(self) -> int:
        return len(self.m)

    @classmethod
    def build(cls, envelopes: list[Envelope]) -> EvpMatrices:
        evp = cls.__new__(cls)
        evp.m = envelopes
        return evp
