
from __future__ import annotations
from .binary import BinaryReader, BinaryWriter
from collections.abc import Callable
from typing import Any, Generic, TypeVar

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

    names = [r.read_key(16) for _ in range(entry_count)]
    return Dictionary(revision, names, data, data_size, tree_bytes)


def write_dictionary(w: BinaryWriter, dict: Dictionary[T], write_data: Callable[[BinaryWriter, T]], data_size=None):
    keys = dict.keys()
    entry_count = len(keys)
    size = dict.data_size if data_size is None else data_size

    start = w.tell()
    w.write_u8(dict.revision)
    w.write_u8(entry_count)
    pos_size = w.tell()
    w.write_u16(0)
    w.write_u16(8)
    w.write_u16((entry_count + 1) * 4 + 8)  # offset to entries

    tree = dict.tree_bytes
    if len(tree) != (entry_count + 1) * 4:
        tree = generate_patricia_tree(keys)
    w.write_bytes(tree)

    w.write_u16(size)
    w.write_u16(4 + size * entry_count)
    for entry in dict.values():
        epos = w.tell()
        write_data(w, entry)
        w.seek(epos + size)

    for key in keys:
        w.write_key(key, 16)

    w.patch_u16(pos_size, w.tell() - start)


def make_dictionary(d: dict[str, T], data_size: int) -> Dictionary[T]:
    return Dictionary(0, list(d.keys()), list(d.values()), data_size)


class Dictionary(Generic[T]):
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

    def __getitem__(self, key) -> T | None:
        try:
            if isinstance(key, str):
                return self.data[self.names.index(key)]
            return self.data[key]
        except ValueError:
            return None

    def keys(self) -> list[str]:
        return self.names

    def values(self) -> list[T]:
        return self.data


def generate_patricia_tree(keys: list[str]) -> bytes:
    root = TreeNode()
    root.refbit = 0x7F
    root.left = root
    root.right = root
    root.idx = 0
    root.key = "\x00" * 16
    nodes = [root]

    for index, key in enumerate(keys):
        _add_patricia_node(nodes, key, index)

    root = nodes[0]
    ordered = _preorder(root)
    pos = {id(n): i for i, n in enumerate(ordered)}

    out = bytearray()
    for node in ordered:
        out.append(node.refbit & 0xFF)
        out.append(pos[id(node.left)] & 0xFF)
        out.append(pos[id(node.right)] & 0xFF)
        out.append(node.idx & 0xFF)

    return bytes(out)


def _preorder(root: TreeNode):
    """Serialize the nodes as a DFS pre-order of the tree"""

    order = [root]
    seen = {id(root)}
    stack = [root.left]

    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue

        order.append(node)
        seen.add(id(node))
        children = []
        if node.left.refbit < node.refbit:
            children.append(node.left)
        if node.right.refbit < node.refbit:
            children.append(node.right)
        stack.extend(reversed(children))
    return order


def _add_patricia_node(nodes: list[TreeNode], key: str, index: int):
    key = key.ljust(16, "\x00")

    leaf = nodes[0].left
    if nodes[0].refbit > leaf.refbit:
        refbit = leaf.refbit
        while True:
            prev = leaf
            leaf = leaf.left if _key_bit(key, refbit) == 0 else leaf.right
            refbit = leaf.refbit
            if prev.refbit <= refbit:
                break

    ref = 0x7F
    if (leaf.idx ^ _string_part(key, 12)) >= 0:
        while True:
            ref -= 1
            chunk = (ref >> 5) & 0x3
            diff = _string_part(
                leaf.key, chunk * 4) ^ _string_part(key, chunk * 4)
            if ((diff >> (ref & 0x1F)) & 1) != 0:
                break

    node5 = nodes[0].left
    node6 = nodes[0]
    refbit1 = node5.refbit
    if node6.refbit > node5.refbit:
        while refbit1 > ref:
            node6 = node5
            node5 = node5.left if _key_bit(key, refbit1) == 0 else node5.right
            refbit1 = node5.refbit
            if node6.refbit <= refbit1:
                break

    node = TreeNode()
    node.refbit = ref
    node.idx = index
    node.key = key
    bit = _key_bit(key, ref)
    node.left = node5 if bit != 0 else node
    node.right = node if bit != 0 else node5

    parent_bit = _key_bit(key, node6.refbit)
    if parent_bit != 0:
        node6.right = node
    else:
        node6.left = node

    nodes.append(node)


def _string_part(s: str, offset: int) -> int:
    part = 0
    for i in range(4):
        part |= ord(s[offset + i]) << (i * 8)
    return part


def _key_bit(key: str, refbit: int) -> int:
    chunk = ((refbit >> 5) & 0x3) * 4
    return (_string_part(key, chunk) >> (refbit & 0x1F)) & 1


class TreeNode:
    __slots__ = ("refbit", "left", "right", "idx", "key")

    def __init__(self):
        self.refbit = 0
        self.left: TreeNode | None = None
        self.right: TreeNode | None = None
        self.idx = 0
        self.key = ""
