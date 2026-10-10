"""Utilitários para formatação e exportação de legendas e transcrições."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Iterable

SUPPORTED_MEDIA_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".flv", ".webm", ".wmv", ".m4v",
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".wma", ".opus"
}


def is_supported_media(file_path: str | Path) -> bool:
    """Verifica se o arquivo possui uma extensão de áudio ou vídeo suportada."""
    path = Path(file_path)
    return path.suffix.lower() in SUPPORTED_MEDIA_EXTENSIONS


def format_timestamp(seconds: float, format_type: str = "srt") -> str:
    """
    Formata um valor em segundos para o formato de tempo de legendas.

    Args:
        seconds: Tempo em segundos (ex: 65.34).
        format_type: 'srt' (ex: 00:01:05,340) ou 'vtt' (ex: 00:01:05.340).

    Returns:
        String formatada.
    """
    if seconds < 0:
        seconds = 0.0

    hours, remainder = divmod(seconds, 3600)
    minutes, remainder = divmod(remainder, 60)
    secs, millis = divmod(remainder, 1)
    millis = int(round(millis * 1000))

    if millis >= 1000:
        secs += 1
        millis -= 1000
    if secs >= 60:
        minutes += 1
        secs -= 60
    if minutes >= 60:
        hours += 1
        minutes -= 60

    separator = "," if format_type.lower() == "srt" else "."
    return f"{int(hours):02}:{int(minutes):02}:{int(secs):02}{separator}{millis:03}"


def write_srt(segments: Iterable[dict[str, Any]], output_path: Path) -> None:
    """Escreve segmentos no padrão SubRip (.srt) com codificação UTF-8."""
    with open(output_path, "w", encoding="utf-8") as f:
        for index, segment in enumerate(segments, start=1):
            start = format_timestamp(segment["start"], "srt")
            end = format_timestamp(segment["end"], "srt")
            text = segment.get("text", "").strip()

            f.write(f"{index}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{text}\n\n")


def write_vtt(segments: Iterable[dict[str, Any]], output_path: Path) -> None:
    """Escreve segmentos no padrão WebVTT (.vtt) com codificação UTF-8."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for index, segment in enumerate(segments, start=1):
            start = format_timestamp(segment["start"], "vtt")
            end = format_timestamp(segment["end"], "vtt")
            text = segment.get("text", "").strip()

            f.write(f"{index}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{text}\n\n")


def write_txt(segments: Iterable[dict[str, Any]], output_path: Path) -> None:
    """Escreve a transcrição em texto puro (.txt), parágrafo por parágrafo."""
    with open(output_path, "w", encoding="utf-8") as f:
        for segment in segments:
            text = segment.get("text", "").strip()
            if text:
                f.write(f"{text}\n")


def write_json(result: dict[str, Any], output_path: Path) -> None:
    """Salva a estrutura completa de resultado do Whisper em JSON."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def export_subtitles(
    result: dict[str, Any],
    base_file_path: Path,
    output_dir: Path | None = None,
    formats: list[str] | None = None
) -> list[Path]:
    """
    Exporta o resultado do Whisper para os formatos solicitados.

    Args:
        result: Dicionário retornado por model.transcribe(...).
        base_file_path: Caminho do arquivo original de mídia.
        output_dir: Diretório de destino (opcional, padrão: diretório do arquivo original).
        formats: Lista de formatos ('srt', 'vtt', 'txt', 'json').

    Returns:
        Lista com os caminhos dos arquivos gerados.
    """
    if formats is None:
        formats = ["srt"]

    out_dir = Path(output_dir) if output_dir else base_file_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    base_stem = base_file_path.stem
    segments = result.get("segments", [])
    created_files: list[Path] = []

    normalized_formats = {fmt.strip().lower() for fmt in formats}
    if "all" in normalized_formats:
        normalized_formats = {"srt", "vtt", "txt", "json"}

    for fmt in normalized_formats:
        out_file = out_dir / f"{base_stem}.{fmt}"
        if fmt == "srt":
            write_srt(segments, out_file)
            created_files.append(out_file)
        elif fmt == "vtt":
            write_vtt(segments, out_file)
            created_files.append(out_file)
        elif fmt == "txt":
            write_txt(segments, out_file)
            created_files.append(out_file)
        elif fmt == "json":
            write_json(result, out_file)
            created_files.append(out_file)

    return created_files
