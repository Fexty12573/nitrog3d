
from .binary import BinaryReader
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar('T')


def read_dictionary(r: BinaryReader, read_data: Callable[[BinaryReader], T]) -> Dictionary[T]:
    revision = r.read_u8()
    entry_count = r.read_u8()
    r.skip(2 * 3)  # sizeof(dict block), padding, offset

    # Patricia tree nodes
    # Only stored so in the case of an unchanged dict size, the tree doesn't
    # have to be regenerated.
    tree_bytes = r.read_bytes((entry_count + 1) * 4)

    data_size = r.read_u16()
    r.skip(2)  # names offset

    data = []
    for _ in range(entry_count):
        start = r.tell()
        data.append(read_data(r))
        r.seek(start + data_size)

    names = [r.read_name(16) for _ in range(entry_count)]
    return Dictionary(revision, names, data, data_size, tree_bytes)


class Dictionary[T]:
    """
    An NSBMD Dictionary
    """

    __slots__ = ("revision", "names", "data", "data_size", "tree_bytes")

    def __init__(self, rev: int, names: list[str], data: list[T], data_size: int, tree_bytes: bytes = b""):
        self.revision = rev
        self.names = names
        self.data = data
        self.data_size = data_size
        self.tree_bytes = tree_bytes

    @property
    def entry_count(self) -> int:
        return len(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> zip[tuple[str, T]]:
        return iter(zip(self.names, self.data))

    def index_of(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError:
            return -1

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.data[self.names.index(key)]
        return self.data[key]

    def keys(self) -> list[str]:
        return self.names

    def values(self) -> list[T]:
        return self.data
