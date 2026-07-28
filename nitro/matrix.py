
import numpy as np


def identity():
    return np.identity(4)


def from4x4(vals: list[float]) -> np.ndarray:
    return np.array(vals).reshape(4, 4)


def from4x3(vals: list[float], col3: list[float] = [0.0, 0.0, 0.0, 1.0]) -> np.ndarray:
    m = np.array(vals).reshape(4, 3)
    m = np.hstack((m, np.array(col3).reshape(4, 1)))
    return m


def from3x3(vals: list[float], col3: list[float] = [0.0, 0.0, 0.0], row3: list[float] = [0.0, 0.0, 0.0, 1.0]) -> np.ndarray:
    m = np.array(vals).reshape(3, 3)
    m = np.hstack((m, np.array(col3).reshape(3, 1)))
    m = np.vstack((m, np.array(row3).reshape(1, 4)))
    return m


def from_rot_trans(rot3x3: np.ndarray, trans: tuple[float, float, float]) -> np.ndarray:
    return from3x3(rot3x3, row3=[*trans, 1.0])


def scale(x: float, y: float, z: float) -> np.ndarray:
    return np.array([[x, 0.0, 0.0, 0.0],
                     [0.0, y, 0.0, 0.0],
                     [0.0, 0.0, z, 0.0],
                     [0.0, 0.0, 0.0, 1.0]])


def translate(x: float, y: float, z: float) -> np.ndarray:
    return np.array([[1.0, 0.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0, 0.0],
                     [x, y, z, 1.0]])


def mul_no_translate(v: tuple[float, float, float], m: np.ndarray) -> tuple[float, float, float]:
    return tuple(np.dot(np.array(v), m[np.ix_([0, 1, 2], [0, 1, 2])]).tolist())


def mul(v: tuple[float, float, float], m: np.ndarray) -> tuple[float, float, float]:
    v4 = np.array([v[0], v[1], v[2], 1.0])
    res = np.dot(v4, m)
    return tuple(res[:3].tolist())


def inverse(m: np.ndarray) -> np.ndarray:
    return np.linalg.inv(m)
