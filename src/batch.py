"""Módulo para processamento em lote (batch) de múltiplos arquivos de mídia."""

from __future__ import annotations
import sys
import time
from pathlib import Path
from typing import Any, Optional

from src.transcriber import WhisperTranscriber
from src.utils import export_subtitles, is_supported_media


def find_media_files(folder: Path, recursive: bool = False) -> list[Path]:
    """
    Encontra todos os arquivos de áudio e vídeo suportados dentro de um diretório.

    Args:
        folder: Caminho da pasta.
        recursive: Se True, busca também em todas as subpastas.

    Returns:
        Lista ordenada de caminhos para os arquivos encontrados.
    """
    if not folder.is_dir():
        raise NotADirectoryError(f"O caminho informado não é um diretório: {folder}")

    pattern = "**/*" if recursive else "*"
    files = [
        f for f in folder.glob(pattern)
        if f.is_file() and is_supported_media(f)
    ]
    return sorted(files)


def process_batch(
    transcriber: WhisperTranscriber,
    folder: Path,
    formats: list[str],
    output_dir: Optional[Path] = None,
    language: Optional[str] = None,
    task: str = "transcribe",
    initial_prompt: Optional[str] = None,
    vad_filter: bool = True,
    recursive: bool = False,
    quiet: bool = False,
) -> dict[str, Any]:
    """
    Processa todos os arquivos de mídia de uma pasta em fila.

    Returns:
        Estatísticas do lote (sucessos, falhas, tempo total).
    """
    media_files = find_media_files(folder, recursive=recursive)

    total_files = len(media_files)
    if total_files == 0:
        print(f"[Lote] Nenhum arquivo de áudio ou vídeo suportado foi encontrado em: {folder}")
        return {"total": 0, "success": 0, "failed": 0, "elapsed_seconds": 0.0}

    print("\n" + "#" * 65)
    print(f"  PROCESSAMENTO EM LOTE INICIADO")
    print(f"  - Pasta: {folder.resolve()}")
    print(f"  - Total de arquivos: {total_files}")
    print(f"  - Busca recursiva: {recursive}")
    print(f"  - Formatos de saída: {', '.join(formats)}")
    print("#" * 65 + "\n")

    batch_start = time.time()
    successful: list[Path] = []
    failed: list[tuple[Path, str]] = []

    for idx, file_path in enumerate(media_files, start=1):
        print("\n" + "-" * 65)
        print(f"  [Fila {idx}/{total_files}] Processando: {file_path.name}")
        print("-" * 65)

        try:
            result = transcriber.transcribe(
                audio_path=file_path,
                language=language,
                task=task,
                initial_prompt=initial_prompt,
                vad_filter=vad_filter,
                verbose=not quiet,
            )

            target_out = output_dir if output_dir else file_path.parent
            created = export_subtitles(
                result=result,
                base_file_path=file_path,
                output_dir=target_out,
                formats=formats,
            )
            print(f"  [OK] Legendas salvas para {file_path.name}:")
            for c in created:
                print(f"    -> {c.name}")
            successful.append(file_path)

        except Exception as exc:
            print(f"  [FALHA] Erro ao processar '{file_path.name}': {exc}", file=sys.stderr)
            failed.append((file_path, str(exc)))

    total_time = time.time() - batch_start

    print("\n" + "=" * 65)
    print(f"  RESUMO DO PROCESSAMENTO EM LOTE")
    print(f"  - Concluídos com sucesso: {len(successful)}/{total_files}")
    if failed:
        print(f"  - Falhas: {len(failed)}")
        for f_path, err in failed:
            print(f"    * {f_path.name}: {err}")
    print(f"  - Tempo total do lote: {total_time:.2f}s ({total_time / 60:.2f} min)")
    print("=" * 65 + "\n")

    return {
        "total": total_files,
        "success": len(successful),
        "failed": len(failed),
        "elapsed_seconds": total_time,
    }
