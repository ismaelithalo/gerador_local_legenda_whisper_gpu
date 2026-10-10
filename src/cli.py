"""Interface de linha de comando (CLI) para o Whisper Subtitle Generator com suporte a Lote e Múltiplos Motores."""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

from src.transcriber import WhisperTranscriber, check_gpu_status
from src.utils import export_subtitles, is_supported_media
from src.batch import process_batch


def parse_arguments(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gerador Local de Legendas e Transcrições com Whisper (GPU NVIDIA CUDA & faster-whisper)",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "file",
        nargs="?",
        help="Caminho do arquivo de áudio ou vídeo, ou diretório para processar em lote.\n"
             "Você também pode arrastar e soltar arquivos/pastas sobre o run.bat.",
    )

    parser.add_argument(
        "--batch", "-b",
        default=None,
        help="Caminho de um diretório contendo múltiplos arquivos de mídia para processar em fila.\n"
             "Exemplo: --batch \"C:\\Videos\\Aulas\"",
    )

    parser.add_argument(
        "--engine", "-e",
        default="faster-whisper",
        choices=["faster-whisper", "openai"],
        help="Motor de inferência a ser utilizado:\n"
             "  faster-whisper (Padrão) - Baseado em CTranslate2: até 4x mais rápido e consome metade da VRAM.\n"
             "  openai                  - Implementação original do OpenAI Whisper via PyTorch.",
    )

    parser.add_argument(
        "--model", "-m",
        default="medium",
        choices=["tiny", "base", "small", "medium", "large-v3", "turbo"],
        help="Modelo do Whisper a ser utilizado:\n"
             "  tiny     - Ultrarrápido, menor consumo de VRAM (~1 GB), menor precisão.\n"
             "  base     - Rápido, leve (~1 GB VRAM).\n"
             "  small    - Bom equilíbrio (~2 GB VRAM).\n"
             "  medium   - Recomendado (Padrão) - Ótima precisão e velocidade (~2.5 GB no faster-whisper).\n"
             "  turbo    - Otimizado para velocidade com qualidade de modelo large (~3 GB no faster-whisper).\n"
             "  large-v3 - Máxima precisão possível (~4-5 GB VRAM no faster-whisper).",
    )

    parser.add_argument(
        "--compute-type",
        default=None,
        choices=["float16", "int8", "int8_float16", "float32"],
        help="Precisão numérica de computação no faster-whisper (padrão: float16 na GPU, int8 na CPU).",
    )

    parser.add_argument(
        "--language", "-l",
        default=None,
        help="Código do idioma do áudio (ex: 'pt', 'ja', 'en', 'es', 'fr', 'zh').\n"
             "Se omitido, o Whisper detectará o idioma automaticamente.",
    )

    parser.add_argument(
        "--task", "-t",
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Modo de operação:\n"
             "  transcribe (Padrão) - Transcreve no mesmo idioma falado no áudio.\n"
             "  translate           - Traduz o áudio para legendas em inglês.",
    )

    parser.add_argument(
        "--format", "-f",
        default="srt",
        help="Formatos de legenda a exportar separados por vírgula (padrão: srt):\n"
             "  srt, vtt, txt, json ou 'all' (para gerar todos simultaneamente).",
    )

    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Diretório onde as legendas serão salvas.\n"
             "Se omitido, os arquivos serão salvos na mesma pasta dos arquivos originais.",
    )

    parser.add_argument(
        "--device", "-d",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Dispositivo de processamento:\n"
             "  auto (Padrão) - Usa GPU dedicada se disponível, senão CPU.\n"
             "  cuda          - Força o uso da GPU NVIDIA CUDA.\n"
             "  cpu           - Força o uso do processador.",
    )

    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Quando em modo de lote (--batch), busca arquivos de mídia também em todas as subpastas.",
    )

    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Desativa o filtro Voice Activity Detection (VAD) do faster-whisper.",
    )

    parser.add_argument(
        "--prompt", "-p",
        default=None,
        help="Prompt inicial para guiar o Whisper com nomes próprios, termos técnicos ou pontuação.",
    )

    parser.add_argument(
        "--check-gpu",
        action="store_true",
        help="Exibe informações sobre a GPU/CUDA detectada e encerra a execução.",
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Desativa a exibição em tempo real dos trechos transcritos.",
    )

    return parser.parse_args(args)


def main(args=None) -> int:
    parsed_args = parse_arguments(args)

    if parsed_args.check_gpu:
        status = check_gpu_status()
        print("\n=== Diagnóstico de Hardware ===")
        print(f"Status GPU: {'Disponível via CUDA' if status['available'] else 'Indisponível (usando CPU)'}")
        print(f"Dispositivo: {status['name']}")
        if status["available"]:
            print(f"VRAM Total: {status['vram_gb']} GB")
            print(f"CUDA Runtime: {status['cuda_version']}")
        print("================================\n")
        return 0

    # Determina se a execução é em lote ou arquivo individual
    target_path = None
    is_batch_mode = False

    if parsed_args.batch:
        target_path = Path(parsed_args.batch).resolve()
        is_batch_mode = True
    elif parsed_args.file:
        candidate = Path(parsed_args.file).resolve()
        if candidate.is_dir():
            target_path = candidate
            is_batch_mode = True
        else:
            target_path = candidate
            is_batch_mode = False
    else:
        print("[Erro] Nenhum arquivo ou diretório foi informado.")
        print("Uso arquivo único: python main.py \"caminho/para/video.mp4\"")
        print("Uso em lote (pasta): python main.py --batch \"caminho/para/pasta\"")
        print("Para ver todas as opções: python main.py --help")
        print("Para testar a GPU: python main.py --check-gpu")
        return 1

    if not target_path.exists():
        print(f"[Erro] Caminho não encontrado: {target_path}")
        return 1

    formats = [f.strip().lower() for f in parsed_args.format.split(",")]
    output_dir = Path(parsed_args.output_dir).resolve() if parsed_args.output_dir else None

    try:
        transcriber = WhisperTranscriber(
            model_name=parsed_args.model,
            engine=parsed_args.engine,
            device=parsed_args.device,
            compute_type=parsed_args.compute_type,
        )

        if is_batch_mode:
            stats = process_batch(
                transcriber=transcriber,
                folder=target_path,
                formats=formats,
                output_dir=output_dir,
                language=parsed_args.language,
                task=parsed_args.task,
                initial_prompt=parsed_args.prompt,
                vad_filter=not parsed_args.no_vad,
                recursive=parsed_args.recursive,
                quiet=parsed_args.quiet,
            )
            return 0 if stats["failed"] == 0 else 1

        else:
            if not is_supported_media(target_path):
                print(f"[Aviso] A extensão '{target_path.suffix}' pode não ser um formato padrão de mídia.")
                print("Tentando prosseguir via ffmpeg mesmo assim...")

            result = transcriber.transcribe(
                audio_path=target_path,
                language=parsed_args.language,
                task=parsed_args.task,
                initial_prompt=parsed_args.prompt,
                vad_filter=not parsed_args.no_vad,
                verbose=not parsed_args.quiet,
            )

            created_files = export_subtitles(
                result=result,
                base_file_path=target_path,
                output_dir=output_dir,
                formats=formats,
            )

            print("\nArquivos de legenda gerados com sucesso:")
            for out_file in created_files:
                print(f" -> {out_file.resolve()}")
            print()
            return 0

    except Exception as exc:
        print(f"\n[Falha durante a execução]: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
