from nitro.dictionary import Dictionary
from nitro.container import Container
from nitro.mdl0 import MDL0
from nitro.nsbmd import NSBMD
from tests.test_mdl0 import build_model
from tests.test_tex0 import build_tex0


def build_container(signature: str) -> Container:
    c = Container.__new__(Container)
    c.signature = signature
    c.endianness = 0xFEFF
    c.version = 2
    c.header_size = 16
    c.file_size = 0
    c.block_offsets = []
    return c


class TestNSBMD:
    def test_bmd0_roundtrips_model_and_texture(self):
        mdl0 = MDL0.__new__(MDL0)
        mdl0.dict = Dictionary(0, ["model0"], [0], 4)
        mdl0.models = [build_model()]

        nsbmd = NSBMD.__new__(NSBMD)
        nsbmd.container = build_container("BMD0")
        nsbmd.model_set = mdl0
        nsbmd.tex_pltt_set = build_tex0()

        data = nsbmd.write()
        out = NSBMD(data)

        assert out.container.signature == "BMD0"
        assert out.model_set.dict.names == ["model0"]
        assert out.model_set.models[0].sbc == mdl0.models[0].sbc
        assert out.tex_pltt_set.tex_dict.names == ["tex0"]
        assert out.tex_pltt_set.tex_dict.values()[0].data == \
            nsbmd.tex_pltt_set.tex_dict.values()[0].data

    def test_bmd0_without_texture_block(self):
        mdl0 = MDL0.__new__(MDL0)
        mdl0.dict = Dictionary(0, ["model0"], [0], 4)
        mdl0.models = [build_model()]

        nsbmd = NSBMD.__new__(NSBMD)
        nsbmd.container = build_container("BMD0")
        nsbmd.model_set = mdl0
        nsbmd.tex_pltt_set = None

        data = nsbmd.write()
        out = NSBMD(data)

        assert out.container.num_blocks == 1
        assert out.model_set.dict.names == ["model0"]
        assert out.tex_pltt_set is None

    def test_btx0_roundtrips_texture_only(self):
        nsbmd = NSBMD.__new__(NSBMD)
        nsbmd.container = build_container("BTX0")
        nsbmd.model_set = None
        nsbmd.tex_pltt_set = build_tex0()

        data = nsbmd.write()
        out = NSBMD(data)

        assert out.container.signature == "BTX0"
        assert out.model_set is None
        assert out.tex_pltt_set.tex_dict.names == ["tex0"]

    def test_file_size_header_field_matches_actual_length(self):
        nsbmd = NSBMD.__new__(NSBMD)
        nsbmd.container = build_container("BTX0")
        nsbmd.model_set = None
        nsbmd.tex_pltt_set = build_tex0()

        data = nsbmd.write()
        out = NSBMD(data)

        assert out.container.file_size == len(data)
