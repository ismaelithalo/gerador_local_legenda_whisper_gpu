"""Testes unitários para utilitários de legendas e formatos."""

import tempfile
import unittest
from pathlib import Path
from src.utils import format_timestamp, export_subtitles, is_supported_media


class TestUtils(unittest.TestCase):

    def test_format_timestamp_srt(self):
        # 0 segundos
        self.assertEqual(format_timestamp(0.0, "srt"), "00:00:00,000")
        # Segundos e milissegundos
        self.assertEqual(format_timestamp(1.234, "srt"), "00:00:01,234")
        # Minutos
        self.assertEqual(format_timestamp(65.5, "srt"), "00:01:05,500")
        # Horas
        self.assertEqual(format_timestamp(3661.050, "srt"), "01:01:01,050")

    def test_format_timestamp_vtt(self):
        self.assertEqual(format_timestamp(65.5, "vtt"), "00:01:05.500")
        self.assertEqual(format_timestamp(3600.0, "vtt"), "01:00:00.000")

    def test_is_supported_media(self):
        self.assertTrue(is_supported_media("video.mp4"))
        self.assertTrue(is_supported_media("audio.mp3"))
        self.assertTrue(is_supported_media("movie.MKV"))
        self.assertTrue(is_supported_media("test.wav"))
        self.assertFalse(is_supported_media("document.pdf"))
        self.assertFalse(is_supported_media("script.py"))

    def test_export_subtitles(self):
        sample_result = {
            "text": "Olá mundo. Teste de transcrição.",
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 2.5,
                    "text": " Olá mundo.",
                },
                {
                    "id": 1,
                    "start": 2.5,
                    "end": 5.0,
                    "text": " Teste de transcrição.",
                }
            ]
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            fake_video = tmp_path / "video_teste.mp4"
            fake_video.touch()

            created = export_subtitles(
                result=sample_result,
                base_file_path=fake_video,
                output_dir=tmp_path,
                formats=["srt", "vtt", "txt", "json"],
            )

            created_names = {f.name for f in created}
            self.assertIn("video_teste.srt", created_names)
            self.assertIn("video_teste.vtt", created_names)
            self.assertIn("video_teste.txt", created_names)
            self.assertIn("video_teste.json", created_names)

            # Validação do conteúdo SRT
            srt_content = (tmp_path / "video_teste.srt").read_text(encoding="utf-8")
            self.assertIn("1\n00:00:00,000 --> 00:00:02,500\nOlá mundo.", srt_content)
            self.assertIn("2\n00:00:02,500 --> 00:00:05,000\nTeste de transcrição.", srt_content)

            # Validação do conteúdo VTT
            vtt_content = (tmp_path / "video_teste.vtt").read_text(encoding="utf-8")
            self.assertTrue(vtt_content.startswith("WEBVTT"))


if __name__ == "__main__":
    unittest.main()
