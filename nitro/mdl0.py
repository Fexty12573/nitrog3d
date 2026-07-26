
from .binary import BinaryReader, BinaryWriter
from .dictionary import read_dictionary, write_dictionary
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
            r.seek(offset)
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

    def hasflag(self, flag: SrtFlag) -> bool:
        return _hasflag(self.flag, flag)


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

    def __len__(self) -> int:
        return len(self.nodes)


class Material:
    def __init__(self, r: BinaryReader):
        self.tag = r.read_u16()
        self.size = r.read_u16()
        self.diff_amb = r.read_u32()
        self.spec_emi = r.read_u32()
        self.poly_attr = r.read_u32()
        self.poly_attr_mask = r.read_u32()
        self.tex_image_param = r.read_u32()
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
        w.write_u32(self.diff_amb)
        w.write_u32(self.spec_emi)
        w.write_u32(self.poly_attr)
        w.write_u32(self.poly_attr_mask)
        w.write_u32(self.tex_image_param)
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

    def hasflag(self, flag: MatFlag) -> bool:
        return _hasflag(self.flag, flag)

    @property
    def tex_width(self) -> int:
        return 8 << ((self.tex_image_param >> 20) & 0x7)

    @property
    def tex_height(self) -> int:
        return 8 << ((self.tex_image_param >> 23) & 0x7)

    @property
    def tex_format(self) -> int:
        return (self.tex_image_param >> 26) & 0x7

    @property
    def cull_mode(self) -> int:
        return (self.poly_attr >> 6) & 0x3

    @property
    def alpha(self) -> int:
        return (self.poly_attr >> 16) & 0x1F


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

    def __len__(self) -> int:
        return len(self.materials)


class TexToMatData:
    """Maps a texture (or palette) name to the material indices that use it"""

    def __init__(self, r: BinaryReader):
        self.offset = r.read_u16()
        self.mat_count = r.read_u8() & 0x7F
        self.flags = r.read_u8()
        self.materials: list[int] = []  # Filled in by MaterialSet

    def write(self, w: BinaryWriter):
        w.write_u16(self.offset)
        w.write_u8(self.mat_count)
        w.write_u8(self.flags)


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
        write_dictionary(w, self.dict, lambda wr, v: wr.write_u32(v))
        for shape in self.shapes:
            shape.write(w)
        w.seek(end)

    def __len__(self) -> int:
        return len(self.shapes)


class Envelope:
    def __init__(self, r: BinaryReader):
        self.inv_m = r.read_fx32s(12)  # Row major 4x3 matrix
        self.inv_n = r.read_fx32s(9)  # Row major 3x3 matrix

    def write(self, w: BinaryWriter):
        w.write_fx32s(self.inv_m)
        w.write_fx32s(self.inv_n)


class EvpMatrices:
    def __init__(self, r: BinaryReader, node_count: int):
        self.m = [Envelope(r) for _ in range(node_count)]

    def write(self, w: BinaryWriter):
        for evp in self.m:
            evp.write(w)

    def __len__(self) -> int:
        return len(self.m)
