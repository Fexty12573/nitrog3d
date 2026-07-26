import struct

from nitro.binary import BinaryReader
from nitro.dictionary import read_dictionary


def make_dict_bytes(entry_payloads: list[bytes], names: list[str], data_size: int, revision: int = 0) -> bytes:
    """Hand-builds the on-disk layout that read_dictionary expects."""
    entry_count = len(entry_payloads)
    buf = struct.pack("<BB", revision, entry_count)
    buf += b"\x00" * 6  # dict block size, padding, offset (unused)
    buf += b"\x00" * ((entry_count + 1) * 4)  # patricia tree nodes
    buf += struct.pack("<H", data_size)
    buf += b"\x00" * 2  # names offset (unused)
    for payload in entry_payloads:
        assert len(payload) <= data_size
        buf += payload + b"\x00" * (data_size - len(payload))
    for name in names:
        raw = name.encode("ascii")
        buf += raw + b"\x00" * (16 - len(raw))
    return buf


class TestReadDictionary:
    def test_reads_header_fields(self):
        raw = make_dict_bytes([struct.pack("<I", 1)],
                              ["only"], data_size=4, revision=5)
        r = BinaryReader(raw)
        d = read_dictionary(r, lambda rd: rd.read_u32())
        assert d.revision == 5
        assert d.data_size == 4
        assert d.entry_count == 1

    def test_reads_names_and_data_in_order(self):
        payloads = [struct.pack("<I", 111), struct.pack(
            "<I", 222), struct.pack("<I", 333)]
        names = ["alpha", "beta", "gamma"]
        r = BinaryReader(make_dict_bytes(payloads, names, data_size=4))
        d = read_dictionary(r, lambda rd: rd.read_u32())
        assert d.names == names
        assert d.data == [111, 222, 333]

    def test_skips_padding_when_data_size_exceeds_payload(self):
        # read_data only consumes 4 bytes, but each entry slot is 8 bytes wide.
        # The reader must still land on the correct offset for the next entry.
        payloads = [struct.pack("<I", 111), struct.pack("<I", 222)]
        r = BinaryReader(make_dict_bytes(payloads, ["a", "b"], data_size=8))
        d = read_dictionary(r, lambda rd: rd.read_u32())
        assert d.data == [111, 222]

    def test_consumes_exactly_its_own_bytes(self):
        raw = make_dict_bytes([struct.pack("<I", 1), struct.pack("<I", 2)], [
                              "a", "b"], data_size=8)
        r = BinaryReader(raw + b"\xff\xff")
        read_dictionary(r, lambda rd: rd.read_u32())
        assert r.tell() == len(raw)

    def test_empty_dictionary(self):
        raw = make_dict_bytes([], [], data_size=4)
        r = BinaryReader(raw)
        d = read_dictionary(r, lambda rd: rd.read_u32())
        assert d.entry_count == 0
        assert d.names == []
        assert d.data == []

    def test_custom_read_data_callback(self):
        payloads = [struct.pack("<HH", 1, 2), struct.pack("<HH", 3, 4)]
        r = BinaryReader(make_dict_bytes(payloads, ["a", "b"], data_size=4))
        d = read_dictionary(r, lambda rd: (rd.read_u16(), rd.read_u16()))
        assert d.data == [(1, 2), (3, 4)]


class TestDictionary:
    def make(self):
        r = BinaryReader(make_dict_bytes(
            [struct.pack("<I", 10), struct.pack("<I", 20)], ["first", "second"], data_size=4))
        return read_dictionary(r, lambda rd: rd.read_u32())

    def test_len(self):
        assert len(self.make()) == 2

    def test_getitem_by_index(self):
        d = self.make()
        assert d[0] == 10
        assert d[1] == 20

    def test_getitem_by_name(self):
        d = self.make()
        assert d["first"] == 10
        assert d["second"] == 20

    def test_index_of(self):
        d = self.make()
        assert d.index_of("second") == 1
        assert d.index_of("missing") == -1

    def test_iter_yields_name_data_pairs(self):
        d = self.make()
        assert list(d) == [("first", 10), ("second", 20)]

    def test_keys_and_values(self):
        d = self.make()
        assert d.keys() == ["first", "second"]
        assert d.values() == [10, 20]
