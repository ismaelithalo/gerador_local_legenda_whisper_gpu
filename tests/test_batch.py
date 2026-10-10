"""Testes unitários para o processamento em lote."""

import tempfile
import unittest
from pathlib import Path
from src.batch import find_media_files


class TestBatch(unittest.TestCase):

    def test_find_media_files_flat(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            (tmp / "video1.mp4").touch()
            (tmp / "audio1.mp3").touch()
            (tmp / "note.txt").touch()
            (tmp / "image.png").touch()

            found = find_media_files(tmp, recursive=False)
            names = [f.name for f in found]

            self.assertEqual(len(names), 2)
            self.assertIn("video1.mp4", names)
            self.assertIn("audio1.mp3", names)
            self.assertNotIn("note.txt", names)

    def test_find_media_files_recursive(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            sub = tmp / "subpasta"
            sub.mkdir()
            (tmp / "video1.mp4").touch()
            (sub / "video2.mkv").touch()

            # Sem recursão deve encontrar apenas 1
            flat_found = find_media_files(tmp, recursive=False)
            self.assertEqual(len(flat_found), 1)

            # Com recursão deve encontrar os 2
            rec_found = find_media_files(tmp, recursive=True)
            self.assertEqual(len(rec_found), 2)
            rec_names = [f.name for f in rec_found]
            self.assertIn("video1.mp4", rec_names)
            self.assertIn("video2.mkv", rec_names)


if __name__ == "__main__":
    unittest.main()
