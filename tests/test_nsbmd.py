from nitro.dictionary import Dictionary
from nitro.mdl0 import MDL0
from nitro.nsbmd import NSBMD, NSBMDHeader
from tests.test_mdl0 import build_model
from tests.test_tex0 import build_tex0


def build_header(signature: str) -> NSBMDHeader:
    h = NSBMDHeader.__new__(NSBMDHeader)
    h.signature = signature
    h.endianness = 0xFEFF
    h.version = 2
    h.header_size = 16
    return h


class TestNSBMD:
    def test_bmd0_roundtrips_model_and_texture(self):
        mdl0 = MDL0.__new__(MDL0)
        mdl0.dict = Dictionary(0, ["model0"], [0], 4)
        mdl0.models = [build_model()]

        nsbmd = NSBMD.__new__(NSBMD)
        nsbmd.header = build_header("BMD0")
        nsbmd.model_set = mdl0
        nsbmd.tex_pltt_set = build_tex0()

        data = nsbmd.write()
        out = NSBMD(data)

        assert out.header.signature == "BMD0"
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
        nsbmd.header = build_header("BMD0")
        nsbmd.model_set = mdl0
        nsbmd.tex_pltt_set = None

        data = nsbmd.write()
        out = NSBMD(data)

        assert out.header.num_blocks == 1
        assert out.model_set.dict.names == ["model0"]
        assert out.tex_pltt_set is None

    def test_btx0_roundtrips_texture_only(self):
        nsbmd = NSBMD.__new__(NSBMD)
        nsbmd.header = build_header("BTX0")
        nsbmd.model_set = None
        nsbmd.tex_pltt_set = build_tex0()

        data = nsbmd.write()
        out = NSBMD(data)

        assert out.header.signature == "BTX0"
        assert out.model_set is None
        assert out.tex_pltt_set.tex_dict.names == ["tex0"]

    def test_file_size_header_field_matches_actual_length(self):
        nsbmd = NSBMD.__new__(NSBMD)
        nsbmd.header = build_header("BTX0")
        nsbmd.model_set = None
        nsbmd.tex_pltt_set = build_tex0()

        data = nsbmd.write()
        out = NSBMD(data)

        assert out.header.file_size == len(data)
