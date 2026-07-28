from pathlib import Path

import numpy as np
import pytest

from nitro.dl import GeometryBuilder
from nitro.nsbmd import NSBMD
from nitro.sbc import SbcInterpreter

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


def built_models(path: Path):
    """Run every model in a file through the SBC interpreter."""
    nsbmd = NSBMD(path.read_bytes())
    if nsbmd.model_set is None:
        pytest.skip("no models in this file")
    for name, model in zip(nsbmd.model_set.dict.names, nsbmd.model_set.models):
        builder = GeometryBuilder()
        interp = SbcInterpreter(model, builder, {})
        interp.run()
        yield name, model, builder, interp


class TestGeometryAgainstModelMetadata:
    """Check interpreted geometry against numbers the files state themselves.

    Everywhere else the expected values are written by hand from a reading of
    the format, so a misreading gets baked into both the code and its tests.
    The counts and bounding box in ModelInfo are independent ground truth: they
    were produced by Nintendo's exporter from the same geometry, so they catch
    exactly the errors a hand-written expectation cannot.
    """

    def test_vertex_and_primitive_counts_match_model_info(self, sample_path):
        for name, model, b, _ in built_models(sample_path):
            info = model.info
            assert b.num_vertices_emitted == info.vertex_count, name
            assert b.num_triangles == info.triangle_count, name
            assert b.num_quads == info.quad_count, name
            assert info.polygon_count == info.triangle_count + info.quad_count

    def test_triangle_list_length_follows_from_the_counts(self, sample_path):
        # quads are split in two on the way out
        for name, model, b, _ in built_models(sample_path):
            expected = model.info.triangle_count + 2 * model.info.quad_count
            assert len(b.triangles) == expected, name

    def test_geometry_lands_inside_the_stored_bounding_box(self, sample_path):
        """The check that pins down matrix composition order end to end.

        Getting the operand order backwards still produces plausible-looking
        geometry for a single-node model, but throws a jointed one wildly out
        of its own bounding box.
        """
        for name, model, b, _ in built_models(sample_path):
            pts = np.array([v.pos for tri in b.triangles for v in tri])
            if not len(pts):
                continue

            info = model.info
            s = info.box_pos_scale
            lo = np.array([info.box_x, info.box_y, info.box_z]) * s
            extent = np.array([info.box_w, info.box_h, info.box_d]) * s
            hi = lo + extent

            # The box is itself stored as fx16, and is free to be a little
            # loose, so allow 1% of the largest side.
            tol = 0.01 * extent.max()
            assert np.all(pts.min(axis=0) >= lo - tol), \
                f"{name}: {pts.min(axis=0)} below {lo}"
            assert np.all(pts.max(axis=0) <= hi + tol), \
                f"{name}: {pts.max(axis=0)} above {hi}"

    def test_geometry_actually_fills_its_bounding_box(self, sample_path):
        # Containment alone would be satisfied by geometry collapsed to a
        # point, which is what a badly wrong transform tends to produce.
        for name, model, b, _ in built_models(sample_path):
            pts = np.array([v.pos for tri in b.triangles for v in tri])
            if not len(pts):
                continue

            info = model.info
            extent = np.array(
                [info.box_w, info.box_h, info.box_d]) * info.box_pos_scale
            span = pts.max(axis=0) - pts.min(axis=0)
            assert np.all(span >= 0.5 * extent), \
                f"{name}: span {span} vs box {extent}"

    def test_every_node_is_visited(self, sample_path):
        for name, model, _, interp in built_models(sample_path):
            assert len(interp.node_seen) == model.info.node_count, name
            assert all(interp.node_seen), \
                f"{name}: unvisited nodes {[i for i, s in enumerate(interp.node_seen) if not s]}"

    def test_draw_calls_partition_the_triangle_list(self, sample_path):
        for name, model, b, interp in built_models(sample_path):
            assert interp.draw_calls, name
            assert len(interp.draw_calls) == model.info.shape_count, name

            cursor = 0
            for call in interp.draw_calls:
                assert call.tri_start == cursor, name
                assert call.tri_end >= call.tri_start, name
                cursor = call.tri_end
            assert cursor == len(b.triangles), name

    def test_every_vertex_is_bound_to_a_real_node(self, sample_path):
        for name, model, b, _ in built_models(sample_path):
            nodes = {v.node for tri in b.triangles for v in tri}
            assert nodes, name
            assert min(nodes) >= 0, name
            assert max(nodes) < model.info.node_count, name
