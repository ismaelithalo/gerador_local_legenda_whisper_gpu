"""Ponto de entrada principal para execução do Whisper Subtitle Generator."""

import sys

# Garante suporte a UTF-8 no console do Windows (evita erros com caracteres de outros idiomas)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.cli import main

if __name__ == "__main__":
    sys.exit(main())
