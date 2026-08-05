"""Integration tests for nitro.model.load.

These run the whole pipeline - container -> MDL0/TEX0 -> SBC -> display lists
-> decoded textures - over the real game files in data/, and check the result
against numbers the files state about themselves. Unit tests elsewhere pin down
each stage; what can only be caught here is the wiring between them.
"""

from pathlib import Path

import numpy as np
import pytest

from nitro import model
from nitro.model import DecodedTexture, ImportedMesh, ImportedSubModel
from nitro.nsbmd import NSBMD
from tests.test_nsbmd import build_container
from tests.test_tex0 import build_tex0

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLE_PATHS = sorted(DATA_DIR.glob("*.nsbmd"))


@pytest.fixture(params=SAMPLE_PATHS, ids=lambda p: p.name)
def sample_path(request) -> Path:
    if not SAMPLE_PATHS:
        pytest.skip("no sample files found in data/")
    return request.param


@pytest.fixture
def loaded(sample_path):
    """(ImportedModel, NSBMD) for one sample, so tests can cross-check."""
    data = sample_path.read_bytes()
    return model.load(data), NSBMD(data)


def submodel_pairs(imported, nsbmd):
    """Pair each ImportedSubModel with the raw Model it came from."""
    return zip(imported.models, [m for _, m in nsbmd.model_set])


def all_meshes(imported):
    for sub in imported.models:
        for mesh in sub.meshes:
            yield sub, mesh


class TestSubModels:
    def test_one_submodel_per_model_in_the_file(self, loaded):
        # Regression: the mesh/material build and the append used to sit inside
        # the per-bone loop, so this came back with one duplicate per bone.
        imported, nsbmd = loaded
        assert len(imported.models) == len(nsbmd.model_set.models)

    def test_submodels_are_distinct_objects(self, loaded):
        imported, _ = loaded
        ids = [id(sub) for sub in imported.models]
        assert len(set(ids)) == len(ids)

    def test_submodel_names_match_the_model_dictionary(self, loaded):
        imported, nsbmd = loaded
        assert [s.name for s in imported.models] == nsbmd.model_set.dict.names

    def test_every_submodel_has_geometry(self, loaded):
        imported, _ = loaded
        for sub in imported.models:
            assert sub.meshes, sub.name
            assert sub.materials, sub.name

    def test_mesh_count_does_not_exceed_shape_count(self, loaded):
        # A shape whose triangles are all degenerate is dropped, so this is an
        # upper bound rather than an equality.
        imported, nsbmd = loaded
        for sub, raw in submodel_pairs(imported, nsbmd):
            assert 0 < len(sub.meshes) <= raw.info.shape_count, sub.name


class TestBones:
    def test_bone_count_matches_model_info(self, loaded):
        imported, nsbmd = loaded
        for sub, raw in submodel_pairs(imported, nsbmd):
            assert len(sub.bones) == raw.info.node_count, sub.name

    def test_bone_names_match_the_node_dictionary(self, loaded):
        imported, nsbmd = loaded
        for sub, raw in submodel_pairs(imported, nsbmd):
            assert [b.name for b in sub.bones] == raw.nodes.dict.names, sub.name

    def test_bone_parents_are_valid_indices(self, loaded):
        imported, _ = loaded
        for sub in imported.models:
            for i, bone in enumerate(sub.bones):
                assert -1 <= bone.parent < len(sub.bones), f"{sub.name} bone {i}"
                assert bone.parent != i, f"{sub.name} bone {i} is its own parent"

    def test_exactly_one_root_bone(self, loaded):
        imported, _ = loaded
        for sub in imported.models:
            roots = [i for i, b in enumerate(sub.bones) if b.parent == -1]
            assert len(roots) == 1, f"{sub.name}: roots {roots}"

    def test_parent_always_precedes_its_child(self, loaded):
        # Not guaranteed by the format, but true of every sample and relied on
        # by consumers that create bones in order (e.g. a Blender armature).
        # If this ever fires, the importer needs a topological pass.
        imported, _ = loaded
        for sub in imported.models:
            for i, bone in enumerate(sub.bones):
                assert bone.parent < i, f"{sub.name} bone {i} parent {bone.parent}"

    def test_hierarchy_is_acyclic(self, loaded):
        imported, _ = loaded
        for sub in imported.models:
            for i in range(len(sub.bones)):
                seen, cur = set(), i
                while cur != -1:
                    assert cur not in seen, f"{sub.name}: cycle at bone {i}"
                    seen.add(cur)
                    cur = sub.bones[cur].parent

    def test_world_matrices_are_4x4(self, loaded):
        imported, _ = loaded
        for sub in imported.models:
            for bone in sub.bones:
                assert isinstance(bone.world_mtx, np.ndarray)
                assert bone.world_mtx.shape == (4, 4), f"{sub.name}/{bone.name}"

    def test_world_matrices_are_finite(self, loaded):
        imported, _ = loaded
        for sub in imported.models:
            for bone in sub.bones:
                assert np.all(np.isfinite(bone.world_mtx)), f"{sub.name}/{bone.name}"


class TestMeshTopology:
    def test_face_indices_are_within_the_vertex_list(self, loaded):
        imported, _ = loaded
        for sub, mesh in all_meshes(imported):
            n = len(mesh.vertices)
            for face in mesh.faces:
                assert all(0 <= i < n for i in face), f"{sub.name}/{mesh.name}: {face}"

    def test_no_degenerate_faces(self, loaded):
        imported, _ = loaded
        for sub, mesh in all_meshes(imported):
            for a, b, c in mesh.faces:
                assert a != b and b != c and a != c, f"{sub.name}/{mesh.name}"

    def test_every_vertex_is_referenced_by_a_face(self, loaded):
        # Vertices are only appended while walking faces, so an orphan means
        # the dedup index and the face list have drifted apart.
        imported, _ = loaded
        for sub, mesh in all_meshes(imported):
            used = {i for face in mesh.faces for i in face}
            assert used == set(range(len(mesh.vertices))), f"{sub.name}/{mesh.name}"

    def test_vertex_bone_is_parallel_to_vertices(self, loaded):
        imported, _ = loaded
        for sub, mesh in all_meshes(imported):
            assert len(mesh.vertex_bone) == len(mesh.vertices), f"{sub.name}/{mesh.name}"

    def test_vertex_bones_reference_real_bones(self, loaded):
        imported, _ = loaded
        for sub, mesh in all_meshes(imported):
            for i, bone in enumerate(mesh.vertex_bone):
                assert 0 <= bone < len(sub.bones), f"{sub.name}/{mesh.name} vertex {i}"

    def test_vertices_are_deduplicated(self, loaded):
        # _build_meshes keys vertices on the rounded position, so no two entries
        # in one mesh may round to the same key.
        imported, _ = loaded
        for sub, mesh in all_meshes(imported):
            keys = [tuple(round(c, 5) for c in v) for v in mesh.vertices]
            assert len(set(keys)) == len(keys), f"{sub.name}/{mesh.name}"

    def test_positions_are_finite(self, loaded):
        imported, _ = loaded
        for sub, mesh in all_meshes(imported):
            assert np.all(np.isfinite(np.array(mesh.vertices))), f"{sub.name}/{mesh.name}"

    @pytest.mark.parametrize("flag,attr,width", [
        ("has_uv", "loop_uvs", 2),
        ("has_normals", "loop_normals", 3),
        ("has_colors", "loop_colors", 3),
    ])
    def test_loop_arrays_are_three_per_face_when_present(self, loaded, flag, attr, width):
        imported, _ = loaded
        for sub, mesh in all_meshes(imported):
            loops = getattr(mesh, attr)
            expected = 3 * len(mesh.faces) if getattr(mesh, flag) else 0
            assert len(loops) == expected, f"{sub.name}/{mesh.name}.{attr}"
            assert all(len(v) == width for v in loops), f"{sub.name}/{mesh.name}.{attr}"

    def test_uvs_are_finite(self, loaded):
        imported, _ = loaded
        for sub, mesh in all_meshes(imported):
            if mesh.has_uv:
                assert np.all(np.isfinite(np.array(mesh.loop_uvs))), f"{sub.name}/{mesh.name}"

    def test_colors_are_normalised(self, loaded):
        imported, _ = loaded
        for sub, mesh in all_meshes(imported):
            for c in mesh.loop_colors:
                assert all(0.0 <= ch <= 1.0 for ch in c), f"{sub.name}/{mesh.name}: {c}"

    def test_material_indices_are_valid(self, loaded):
        imported, _ = loaded
        for sub, mesh in all_meshes(imported):
            assert 0 <= mesh.material < len(sub.materials), \
                f"{sub.name}/{mesh.name}: {mesh.material}"


class TestGeometryAgainstModelInfo:
    """Cross-checks against ground truth the exporter wrote into the file.

    Hand-written expectations can bake in the same misreading as the code;
    ModelInfo cannot.
    """

    def test_total_faces_match_the_stored_primitive_counts(self, loaded):
        imported, nsbmd = loaded
        for sub, raw in submodel_pairs(imported, nsbmd):
            info = raw.info
            # quads are split into two triangles on the way out
            expected = info.triangle_count + 2 * info.quad_count
            actual = sum(len(m.faces) for m in sub.meshes)
            assert actual == expected, sub.name

    def test_vertices_land_inside_the_stored_bounding_box(self, loaded):
        # The end-to-end check on transform composition: a wrong matrix order
        # still yields plausible geometry for a single-node model but throws a
        # jointed one out of its own box.
        imported, nsbmd = loaded
        for sub, raw in submodel_pairs(imported, nsbmd):
            pts = np.array([v for m in sub.meshes for v in m.vertices])
            if not len(pts):
                continue

            info = raw.info
            s = info.box_pos_scale
            lo = np.array([info.box_x, info.box_y, info.box_z]) * s
            extent = np.array([info.box_w, info.box_h, info.box_d]) * s
            tol = 0.01 * extent.max()  # the box is fx16 and a little loose

            assert np.all(pts.min(axis=0) >= lo - tol), \
                f"{sub.name}: {pts.min(axis=0)} below {lo}"
            assert np.all(pts.max(axis=0) <= lo + extent + tol), \
                f"{sub.name}: {pts.max(axis=0)} above {lo + extent}"

    def test_geometry_fills_its_bounding_box(self, loaded):
        # Containment alone is satisfied by geometry collapsed to a point.
        imported, nsbmd = loaded
        for sub, raw in submodel_pairs(imported, nsbmd):
            pts = np.array([v for m in sub.meshes for v in m.vertices])
            if not len(pts):
                continue
            info = raw.info
            extent = np.array([info.box_w, info.box_h, info.box_d]) * info.box_pos_scale
            span = pts.max(axis=0) - pts.min(axis=0)
            assert np.all(span >= 0.5 * extent), f"{sub.name}: {span} vs {extent}"

    def test_material_count_matches_model_info(self, loaded):
        imported, nsbmd = loaded
        for sub, raw in submodel_pairs(imported, nsbmd):
            assert len(sub.materials) == raw.info.mat_count, sub.name


class TestMaterials:
    def test_material_names_match_the_material_dictionary(self, loaded):
        imported, nsbmd = loaded
        for sub, raw in submodel_pairs(imported, nsbmd):
            assert [m.name for m in sub.materials] == raw.materials.dict.names, sub.name

    def test_alpha_is_normalised(self, loaded):
        imported, _ = loaded
        for sub in imported.models:
            for m in sub.materials:
                assert 0.0 <= m.alpha <= 1.0, f"{sub.name}/{m.name}: {m.alpha}"

    def test_diffuse_is_normalised(self, loaded):
        imported, _ = loaded
        for sub in imported.models:
            for m in sub.materials:
                assert len(m.diffuse) == 3
                assert all(0.0 <= c <= 1.0 for c in m.diffuse), f"{sub.name}/{m.name}"

    def test_cull_mode_is_two_bits(self, loaded):
        imported, _ = loaded
        for sub in imported.models:
            for m in sub.materials:
                assert 0 <= m.polygon_attr.cull_mode <= 3, f"{sub.name}/{m.name}"


class TestTextures:
    def test_rgba_length_matches_dimensions(self, loaded):
        imported, _ = loaded
        for name, tex in imported.textures.items():
            assert len(tex.rgba) == tex.width * tex.height * 4, name

    def test_dimensions_are_positive_powers_of_two(self, loaded):
        imported, _ = loaded
        for name, tex in imported.textures.items():
            assert tex.width > 0 and tex.height > 0, name
            assert tex.width & (tex.width - 1) == 0, f"{name}: w={tex.width}"
            assert tex.height & (tex.height - 1) == 0, f"{name}: h={tex.height}"

    def test_texture_keys_match_their_stored_name(self, loaded):
        imported, _ = loaded
        for name, tex in imported.textures.items():
            assert tex.name == name

    def test_every_material_texture_is_in_the_texture_table(self, loaded):
        imported, _ = loaded
        for sub in imported.models:
            for m in sub.materials:
                if m.texture is not None:
                    assert m.texture.name in imported.textures, f"{sub.name}/{m.name}"

    def test_materials_share_one_decoded_texture_object(self, loaded):
        # MaterialBuilder caches on (texture, palette), so two materials using
        # the same pair must get the identical object, not a second decode.
        imported, _ = loaded
        for sub in imported.models:
            for m in sub.materials:
                if m.texture is not None:
                    assert m.texture is imported.textures[m.texture.name], \
                        f"{sub.name}/{m.name}"

    def test_no_textures_when_the_file_has_no_texture_block(self, loaded):
        imported, nsbmd = loaded
        if nsbmd.tex_pltt_set is not None:
            pytest.skip("this sample has a texture block")
        assert imported.textures == {}
        for sub in imported.models:
            assert all(m.texture is None for m in sub.materials), sub.name

    def test_textures_are_decoded_when_the_file_has_a_texture_block(self, loaded):
        imported, nsbmd = loaded
        if nsbmd.tex_pltt_set is None:
            pytest.skip("this sample has no texture block")
        assert imported.textures
        assert any(m.texture is not None
                   for sub in imported.models for m in sub.materials)


class TestDeterminism:
    def test_loading_twice_produces_the_same_result(self, sample_path):
        data = sample_path.read_bytes()
        a, b = model.load(data), model.load(data)

        def summary(r):
            return (
                [s.name for s in r.models],
                [[bo.name for bo in s.bones] for s in r.models],
                [[(m.name, len(m.vertices), len(m.faces), m.material)
                  for m in s.meshes] for s in r.models],
                sorted(r.textures),
            )

        assert summary(a) == summary(b)

    def test_load_does_not_mutate_the_input(self, sample_path):
        data = sample_path.read_bytes()
        before = bytes(data)
        model.load(data)
        assert data == before

    def test_bytearray_input_matches_bytes_input(self, sample_path):
        data = sample_path.read_bytes()
        a = model.load(data)
        b = model.load(bytearray(data))
        assert [s.name for s in a.models] == [s.name for s in b.models]
        assert sorted(a.textures) == sorted(b.textures)


class TestEmptyAndEdgeCases:
    def test_dataclass_defaults_are_not_shared_between_instances(self):
        # A mutable default would make every instance alias one list; with
        # slots=True the class also has to declare these as real fields.
        a, b = ImportedSubModel("a"), ImportedSubModel("b")
        a.bones.append(None)
        assert b.bones == []

        m1, m2 = ImportedMesh("m1"), ImportedMesh("m2")
        m1.vertices.append((0.0, 0.0, 0.0))
        assert m2.vertices == []
        m1.material = 3  # must be a real field, not a bare class attribute
        assert m2.material == -1

    def test_texture_only_file_yields_no_models(self):
        nsbmd = NSBMD.__new__(NSBMD)
        nsbmd.container = build_container("BTX0")
        nsbmd.model_set = None
        nsbmd.tex_pltt_set = build_tex0()

        result = model.load(nsbmd.write())
        assert result.models == []
        assert result.textures == {}
        assert result.anims == []

    def test_decoded_texture_is_hashable_by_name_lookup(self, loaded):
        imported, _ = loaded
        for name, tex in imported.textures.items():
            assert isinstance(tex, DecodedTexture)
            assert isinstance(name, str)
