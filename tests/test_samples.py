from pathlib import Path

import pytest

from nitro.nsbmd import NSBMD

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLE_PATHS = sorted(DATA_DIR.glob("*.nsbmd")) + sorted(DATA_DIR.glob("*.nsbtx"))


def _id(path: Path) -> str:
    return path.name


@pytest.fixture(params=SAMPLE_PATHS, ids=_id)
def sample_path(request) -> Path:
    if not SAMPLE_PATHS:
        pytest.skip("no sample files found in data/")
    return request.param


class TestRealSampleFiles:
    """
    End-to-end tests against real, game-extracted NSBMD/NSBTX files in data/.
    These exist alongside the synthetic-fixture tests in test_mdl0.py/test_tex0.py/
    test_nsbmd.py, which stay useful for pinpointing failures to a specific class
    without needing a real file that happens to exercise that code path.
    """

    def test_parses_without_error(self, sample_path):
        NSBMD(sample_path.read_bytes())

    def test_write_reproduces_original_bytes_exactly(self, sample_path):
        data = sample_path.read_bytes()
        nsbmd = NSBMD(data)
        assert nsbmd.write() == data

    def test_rewritten_output_reparses_to_the_same_structure(self, sample_path):
        data = sample_path.read_bytes()
        nsbmd = NSBMD(data)
        reparsed = NSBMD(nsbmd.write())

        assert reparsed.container.signature == nsbmd.container.signature
        assert reparsed.container.num_blocks == nsbmd.container.num_blocks

        if nsbmd.model_set is not None:
            assert reparsed.model_set.dict.names == nsbmd.model_set.dict.names
            for orig, new in zip(nsbmd.model_set.models, reparsed.model_set.models):
                assert new.sbc == orig.sbc
                assert new.nodes.dict.names == orig.nodes.dict.names
                assert new.materials.dict.names == orig.materials.dict.names
                assert [s.dl for s in new.shapes.shapes] == [s.dl for s in orig.shapes.shapes]

        if nsbmd.tex_pltt_set is not None:
            assert reparsed.tex_pltt_set.tex_dict.names == nsbmd.tex_pltt_set.tex_dict.names
            assert reparsed.tex_pltt_set.pltt_dict.names == nsbmd.tex_pltt_set.pltt_dict.names
            for orig, new in zip(nsbmd.tex_pltt_set.tex_dict.values(),
                                  reparsed.tex_pltt_set.tex_dict.values()):
                assert new.data == orig.data
            for orig, new in zip(nsbmd.tex_pltt_set.pltt_dict.values(),
                                  reparsed.tex_pltt_set.pltt_dict.values()):
                assert new.data == orig.data

    def test_rewriting_twice_is_idempotent(self, sample_path):
        # Guards against a write() that happens to match the original file but
        # drifts on a second pass (e.g. state mutated during the first write).
        nsbmd = NSBMD(sample_path.read_bytes())
        first = nsbmd.write()
        second = NSBMD(first).write()
        assert first == second
