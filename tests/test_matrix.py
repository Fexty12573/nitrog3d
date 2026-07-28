"""Convention tests for nitro.matrix.

Everything here exists to ensure that matrix operations are row-major.
"""

import numpy as np
import pytest

from nitro import matrix as mat

ROT_Z90 = [0, 1, 0, -1, 0, 0, 0, 0, 1]  # 90 deg about Z, row-major


class TestConstructors:
    def test_identity_is_4x4(self):
        m = mat.identity()
        assert m.shape == (4, 4)
        assert m == pytest.approx(np.identity(4))

    def test_from4x4_is_row_major(self):
        m = mat.from4x4(list(range(16)))
        assert m[0].tolist() == [0, 1, 2, 3]
        assert m[3].tolist() == [12, 13, 14, 15]

    def test_from4x3_fills_the_fourth_column(self):
        m = mat.from4x3([0] * 12, col3=[1, 2, 3, 4])
        assert m[:, 3].tolist() == [1, 2, 3, 4]

    def test_from4x3_puts_the_fourth_row_of_values_in_the_bottom_row(self):
        m = mat.from4x3([1, 0, 0, 0, 1, 0, 0, 0, 1, 7, 8, 9])
        assert m[3, :3].tolist() == [7, 8, 9]

    def test_from4x3_default_column_is_affine(self):
        m = mat.from4x3([0] * 12)
        assert m[:, 3].tolist() == [0, 0, 0, 1]

    def test_from3x3_col3_becomes_a_column_and_row3_becomes_a_row(self):
        # The parameter names must match where the values actually land.
        m = mat.from3x3([0] * 9, col3=[7, 7, 7], row3=[9, 9, 9, 9])
        assert m[:3, 3].tolist() == [7, 7, 7]
        assert m[3, :].tolist() == [9, 9, 9, 9]

    def test_from3x3_defaults_are_affine(self):
        m = mat.from3x3([0] * 9)
        assert m[:3, 3].tolist() == [0, 0, 0]
        assert m[3, :].tolist() == [0, 0, 0, 1]

    def test_from3x3_preserves_the_rotation_block(self):
        m = mat.from3x3(ROT_Z90)
        assert m[:3, :3].flatten().tolist() == ROT_Z90


class TestTranslationConvention:
    """Every builder must agree on where translation lives: the bottom row."""

    def test_translate_puts_translation_in_the_bottom_row(self):
        m = mat.translate(1.0, 2.0, 3.0)
        assert m[3, :3].tolist() == [1.0, 2.0, 3.0]

    def test_translate_matches_from4x3(self):
        assert mat.translate(1.0, 2.0, 3.0) == pytest.approx(
            mat.from4x3([1, 0, 0, 0, 1, 0, 0, 0, 1, 1.0, 2.0, 3.0]))

    def test_from_rot_trans_matches_from4x3(self):
        trans = (1.0, 2.0, 3.0)
        assert mat.from_rot_trans(np.identity(3), trans) == pytest.approx(
            mat.from4x3([1, 0, 0, 0, 1, 0, 0, 0, 1, *trans]))

    def test_from_rot_trans_keeps_the_rotation_block(self):
        rot = np.array(ROT_Z90).reshape(3, 3)
        m = mat.from_rot_trans(rot, (1.0, 2.0, 3.0))
        assert m[:3, :3].flatten().tolist() == ROT_Z90
        assert m[3, :3].tolist() == [1.0, 2.0, 3.0]

    def test_from_rot_trans_is_affine(self):
        m = mat.from_rot_trans(np.identity(3), (1.0, 2.0, 3.0))
        assert m[:, 3].tolist() == [0.0, 0.0, 0.0, 1.0]

    @pytest.mark.parametrize("build", [
        pytest.param(lambda t: mat.translate(*t), id="translate"),
        pytest.param(lambda t: mat.from4x3(
            [1, 0, 0, 0, 1, 0, 0, 0, 1, *t]), id="from4x3"),
        pytest.param(lambda t: mat.from_rot_trans(
            np.identity(3), t), id="from_rot_trans"),
        pytest.param(lambda t: mat.from3x3([1, 0, 0, 0, 1, 0, 0, 0, 1], row3=[*t, 1.0]),
                     id="from3x3"),
        pytest.param(lambda t: mat.from4x4([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, *t, 1.0]),
                     id="from4x4"),
    ])
    def test_every_builder_moves_the_origin_the_same_way(self, build):
        assert mat.mul((0.0, 0.0, 0.0), build((1.0, 2.0, 3.0))) == \
            pytest.approx((1.0, 2.0, 3.0))


class TestMul:
    def test_mul_applies_translation(self):
        m = mat.translate(1.0, 2.0, 3.0)
        assert mat.mul((0.0, 0.0, 0.0), m) == pytest.approx((1.0, 2.0, 3.0))

    def test_mul_applies_rotation(self):
        m = mat.from3x3(ROT_Z90)
        assert mat.mul((1.0, 0.0, 0.0), m) == pytest.approx((0.0, 1.0, 0.0))

    def test_mul_applies_rotation_then_translation(self):
        m = mat.from_rot_trans(
            np.array(ROT_Z90).reshape(3, 3), (10.0, 0.0, 0.0))
        assert mat.mul((1.0, 0.0, 0.0), m) == pytest.approx((10.0, 1.0, 0.0))

    def test_mul_returns_three_components(self):
        assert len(mat.mul((1.0, 2.0, 3.0), np.identity(4))) == 3

    def test_identity_is_a_no_op(self):
        assert mat.mul((1.0, 2.0, 3.0), np.identity(
            4)) == pytest.approx((1.0, 2.0, 3.0))

    def test_scale_scales_each_axis(self):
        assert mat.mul((1.0, 1.0, 1.0), mat.scale(2.0, 3.0, 4.0)) == \
            pytest.approx((2.0, 3.0, 4.0))

    def test_matmul_composes_left_to_right(self):
        # v @ (A @ B) == (v @ A) @ B, so A is applied before B
        a = mat.translate(1.0, 0.0, 0.0)
        b = mat.scale(2.0, 2.0, 2.0)
        composed = mat.mul((0.0, 0.0, 0.0), np.matmul(a, b))
        stepwise = mat.mul(mat.mul((0.0, 0.0, 0.0), a), b)
        assert composed == pytest.approx(stepwise)
        assert composed == pytest.approx((2.0, 0.0, 0.0))

    def test_translation_order_matters(self):
        t = mat.translate(1.0, 0.0, 0.0)
        s = mat.scale(2.0, 2.0, 2.0)
        assert mat.mul((0.0, 0.0, 0.0), np.matmul(
            t, s)) == pytest.approx((2.0, 0.0, 0.0))
        assert mat.mul((0.0, 0.0, 0.0), np.matmul(
            s, t)) == pytest.approx((1.0, 0.0, 0.0))


class TestMulNoTranslate:
    def test_ignores_translation(self):
        m = mat.translate(9.0, 9.0, 9.0)
        assert mat.mul_no_translate(
            (1.0, 2.0, 3.0), m) == pytest.approx((1.0, 2.0, 3.0))

    def test_applies_rotation(self):
        m = mat.from3x3(ROT_Z90)
        assert mat.mul_no_translate(
            (1.0, 0.0, 0.0), m) == pytest.approx((0.0, 1.0, 0.0))

    def test_matches_mul_when_there_is_no_translation(self):
        m = mat.from3x3(ROT_Z90)
        v = (0.5, -0.25, 1.0)
        assert mat.mul_no_translate(v, m) == pytest.approx(mat.mul(v, m))

    def test_returns_three_components(self):
        assert len(mat.mul_no_translate((1.0, 2.0, 3.0), np.identity(4))) == 3


class TestInverse:
    def test_round_trips_a_point(self):
        m = mat.from_rot_trans(
            np.array(ROT_Z90).reshape(3, 3), (1.0, 2.0, 3.0))
        v = (0.5, -1.5, 2.0)
        assert mat.mul(mat.mul(v, m), mat.inverse(m)) == pytest.approx(v)

    def test_inverse_of_translation_negates_it(self):
        assert mat.inverse(mat.translate(1.0, 2.0, 3.0)) == pytest.approx(
            mat.translate(-1.0, -2.0, -3.0))
