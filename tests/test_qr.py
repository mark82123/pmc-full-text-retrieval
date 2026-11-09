"""ir/qr.py - the module matrix was cross-checked against the segno library (all 8 masks, versions 1-10, L and M)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ir.qr import _encode_codewords, _rs_remainder, qr_matrix, qr_svg  # noqa: E402


class QRTest(unittest.TestCase):
    def test_reed_solomon_known_vector(self):
        # ISO 18004 example (1-M "HELLO WORLD"): 16 data codewords -> 10 EC codewords
        data = [32, 91, 11, 120, 209, 114, 220, 77, 67, 64, 236, 17, 236, 17, 236, 17]
        self.assertEqual(_rs_remainder(data, 10), [196, 35, 39, 119, 235, 215, 231, 226, 93, 23])

    def test_byte_mode_header_and_padding(self):
        version, cw = _encode_codewords(b"A", "M")
        self.assertEqual(version, 1)
        self.assertEqual(cw[:5], [0x40, 0x14, 0x10, 0xEC, 0x11])   # mode 0100, count 1, 'A', pad codewords
        self.assertEqual(len(cw), 26)

    def test_version_selection(self):
        self.assertEqual(_encode_codewords(b"x" * 14, "M")[0], 1)
        self.assertEqual(_encode_codewords(b"x" * 15, "M")[0], 2)
        self.assertEqual(_encode_codewords(b"x" * 210, "M")[0], 10)
        with self.assertRaises(ValueError):
            _encode_codewords(b"x" * 300, "M")

    def test_matrix_structure(self):
        m = qr_matrix("http://192.168.1.10:8090/")
        n = len(m)
        self.assertEqual(n, 25)                          # version 2 (25 bytes)
        self.assertTrue(all(len(r) == n for r in m))
        finder = ["#######", "#.....#", "#.###.#", "#.###.#", "#.###.#", "#.....#", "#######"]
        for (r0, c0) in ((0, 0), (0, n - 7), (n - 7, 0)):
            got = ["".join("#" if m[r0 + i][c0 + j] else "." for j in range(7)) for i in range(7)]
            self.assertEqual(got, finder)
        self.assertTrue(m[n - 8][8])                     # the dark module
        self.assertTrue(all(m[6][c] == (c % 2 == 0) for c in range(8, n - 8)))   # timing pattern

    def test_masks_differ_only_in_data_area(self):
        a, b = qr_matrix("hello", "M", 0), qr_matrix("hello", "M", 3)
        self.assertNotEqual(a, b)
        self.assertEqual([r[:8] for r in a[:8]], [r[:8] for r in b[:8]])   # finder + separator untouched

    def test_svg(self):
        s = qr_svg("http://10.0.0.1:8090/")
        self.assertTrue(s.startswith("<svg"))
        self.assertIn('viewBox="0 0 264 264"', s)        # (25 + 8) * 8
        self.assertIn("<path", s)


if __name__ == "__main__":
    unittest.main()
