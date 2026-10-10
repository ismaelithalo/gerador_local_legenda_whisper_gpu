"""Interface de linha de comando (CLI) para o Whisper Subtitle Generator."""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

from src.transcriber import WhisperTranscriber, check_gpu_status
from src.utils import export_subtitles, is_supported_media


def parse_arguments(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gerador Local de Legendas e Transcrições com OpenAI Whisper (Acelerado por GPU NVIDIA CUDA)",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "file",
        nargs="?",
        help="Caminho do arquivo de áudio ou vídeo (ex: video.mp4, audio.mp3).\n"
             "Você também pode arrastar e soltar o arquivo no terminal ou sobre o run.bat.",
    )

    parser.add_argument(
        "--model", "-m",
        default="medium",
        choices=["tiny", "base", "small", "medium", "large-v3", "turbo"],
        help="Modelo do Whisper a ser utilizado:\n"
             "  tiny     - Ultrarrápido, menor consumo de VRAM (~1 GB), menor precisão.\n"
             "  base     - Rápido, leve (~1 GB VRAM).\n"
             "  small    - Bom equilíbrio (~2 GB VRAM).\n"
             "  medium   - Recomendado (Padrão) - Ótima precisão e velocidade (~5 GB VRAM).\n"
             "  turbo    - Modelo otimizado para velocidade sem perder qualidade (~6 GB VRAM).\n"
             "  large-v3 - Máxima precisão (~7-8 GB VRAM, perfeito para a RTX 5060 8GB).",
    )

    parser.add_argument(
        "--language", "-l",
        default=None,
        help="Código do idioma do áudio (ex: 'pt', 'ja', 'en', 'es', 'fr', 'zh').\n"
             "Se não for informado, o Whisper detectará o idioma automaticamente.",
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
             "  srt, vtt, txt, json ou 'all' (para gerar todos de uma vez).",
    )

    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Diretório onde as legendas serão salvas.\n"
             "Se omitido, os arquivos serão salvos na mesma pasta do arquivo original.",
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

    if not parsed_args.file:
        print("[Erro] Nenhum arquivo foi informado.")
        print("Uso básico: python main.py \"caminho/para/arquivo.mp4\"")
        print("Para ver todas as opções: python main.py --help")
        print("Para checar a GPU: python main.py --check-gpu")
        return 1

    file_path = Path(parsed_args.file).resolve()
    if not file_path.exists():
        print(f"[Erro] Arquivo não encontrado: {file_path}")
        return 1

    if not is_supported_media(file_path):
        print(f"[Aviso] A extensão '{file_path.suffix}' pode não ser um formato padrão de mídia suportado.")
        print("Tentando prosseguir via ffmpeg mesmo assim...")

    # Formatos solicitados
    formats = [f.strip().lower() for f in parsed_args.format.split(",")]

    try:
        transcriber = WhisperTranscriber(
            model_name=parsed_args.model,
            device=parsed_args.device,
        )

        result = transcriber.transcribe(
            audio_path=file_path,
            language=parsed_args.language,
            task=parsed_args.task,
            initial_prompt=parsed_args.prompt,
            verbose=not parsed_args.quiet,
        )

        output_dir = Path(parsed_args.output_dir) if parsed_args.output_dir else None
        created_files = export_subtitles(
            result=result,
            base_file_path=file_path,
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
