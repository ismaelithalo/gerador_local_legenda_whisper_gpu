"""Módulo de transcrição com suporte a múltiplos motores (faster-whisper e openai-whisper) acelerados por GPU CUDA."""

from __future__ import annotations
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

from src.utils import format_timestamp


def _setup_cuda_dlls():
    """Configura os diretórios de DLLs do CUDA para que o CTranslate2 / faster-whisper encontre cublas e cudnn no Windows."""
    venv_site = os.path.join(sys.prefix, "Lib", "site-packages")
    dll_dirs = [
        os.path.join(venv_site, "nvidia", "cublas", "bin"),
        os.path.join(venv_site, "nvidia", "cudnn", "bin"),
        os.path.join(venv_site, "torch", "lib"),
    ]
    for d in dll_dirs:
        if os.path.isdir(d):
            if d not in os.environ.get("PATH", ""):
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(d)
                except Exception:
                    pass


def _patch_pyav():
    """Compatibiliza o PyAV 19+ com o faster-whisper (removendo parâmetro metadata_errors descontinuado)."""
    try:
        import av
        if not hasattr(av, "_patched_open"):
            _orig_open = av.open

            def _safe_open(*args, **kwargs):
                kwargs.pop("metadata_errors", None)
                return _orig_open(*args, **kwargs)

            av.open = _safe_open
            av._patched_open = True
    except ImportError:
        pass


# Aplica correções de ambiente no carregamento do módulo
_setup_cuda_dlls()
_patch_pyav()


def check_gpu_status() -> dict[str, Any]:
    """
    Verifica a disponibilidade de GPU dedicada via PyTorch CUDA ou CTranslate2.
    Retorna informações detalhadas sobre a GPU ou CPU.
    """
    try:
        import torch
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            device_name = torch.cuda.get_device_name(0)
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            cuda_version = torch.version.cuda
            return {
                "available": True,
                "device": "cuda",
                "name": device_name,
                "vram_gb": round(total_vram, 2),
                "cuda_version": cuda_version,
            }
    except ImportError:
        pass

    try:
        import ctranslate2
        cuda_count = ctranslate2.get_cuda_device_count()
        if cuda_count > 0:
            return {
                "available": True,
                "device": "cuda",
                "name": "GPU NVIDIA compatível (CUDA)",
                "vram_gb": 8.0,
                "cuda_version": "12.x / 13.x",
            }
    except ImportError:
        pass

    return {
        "available": False,
        "device": "cpu",
        "name": "Processador (CPU)",
        "vram_gb": 0.0,
        "cuda_version": None,
    }


class WhisperTranscriber:
    """Gerencia a transcrição de áudio e vídeo suportando os motores faster-whisper e openai-whisper."""

    def __init__(
        self,
        model_name: str = "medium",
        engine: str = "faster-whisper",
        device: str = "auto",
        compute_type: Optional[str] = None,
        download_root: Optional[str | Path] = None,
    ):
        """
        Inicializa o transcritor.

        Args:
            model_name: tiny, base, small, medium, large-v3, turbo
            engine: 'faster-whisper' (padrão, 4x mais rápido) ou 'openai'
            device: 'auto', 'cuda', ou 'cpu'
            compute_type: 'float16', 'int8', 'int8_float16', 'float32'
            download_root: Diretório onde o cache dos modelos será salvo
        """
        self.model_name = model_name
        self.engine = engine.lower().strip()
        self.download_root = str(download_root) if download_root else None
        self.gpu_info = check_gpu_status()
        self.device = self._resolve_device(device)
        self.compute_type = self._resolve_compute_type(compute_type)
        self.model = None

    def _resolve_device(self, requested_device: str) -> str:
        req = requested_device.lower().strip()
        if req == "auto":
            if self.gpu_info["available"]:
                print(f"[Hardware] GPU dedicada detectada: {self.gpu_info['name']} ({self.gpu_info['vram_gb']} GB VRAM)")
                return "cuda"
            else:
                print("[Hardware] Nenhuma GPU CUDA detectada. Utilizando CPU.")
                return "cpu"
        elif req == "cuda":
            if not self.gpu_info["available"]:
                print("[Aviso] GPU CUDA solicitada, mas não está disponível. Revertendo para CPU.")
                return "cpu"
            return "cuda"
        elif req == "cpu":
            return "cpu"
        else:
            raise ValueError(f"Dispositivo desconhecido: {requested_device}. Use 'auto', 'cuda' ou 'cpu'.")

    def _resolve_compute_type(self, compute_type: Optional[str]) -> str:
        if compute_type:
            return compute_type
        if self.device == "cuda":
            return "float16"
        return "int8"

    def load_model(self):
        """Carrega os pesos do modelo Whisper na memória / VRAM."""
        if self.model is not None:
            return self.model

        start_time = time.time()

        if self.engine in ("faster-whisper", "faster", "ctranslate2"):
            try:
                from faster_whisper import WhisperModel
            except ImportError:
                raise ImportError(
                    "O pacote 'faster-whisper' não está instalado.\n"
                    "Execute: pip install faster-whisper"
                )

            print(f"\n[Whisper] Carregando modelo '{self.model_name}' com faster-whisper em [{self.device.upper()} / {self.compute_type}]...")
            self.model = WhisperModel(
                model_size_or_path=self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                download_root=self.download_root,
            )

        else:
            try:
                import whisper
            except ImportError:
                raise ImportError(
                    "O pacote 'openai-whisper' não está instalado.\n"
                    "Execute: pip install openai-whisper"
                )

            print(f"\n[Whisper] Carregando modelo '{self.model_name}' com openai-whisper em [{self.device.upper()}]...")
            self.model = whisper.load_model(
                name=self.model_name,
                device=self.device,
                download_root=self.download_root,
            )

        elapsed = time.time() - start_time
        print(f"[Whisper] Modelo carregado com sucesso em {elapsed:.2f}s!")
        return self.model

    def transcribe(
        self,
        audio_path: str | Path,
        language: Optional[str] = None,
        task: str = "transcribe",
        initial_prompt: Optional[str] = None,
        vad_filter: bool = True,
        temperature: float = 0.0,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """
        Executa a transcrição do arquivo de áudio ou vídeo.

        Returns:
            Dicionário com o resultado padronizado ('text', 'segments', 'language').
        """
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {audio_file.resolve()}")

        model = self.load_model()
        use_fp16 = (self.device == "cuda")

        print(f"\n" + "=" * 60)
        print(f"Iniciando transcrição:")
        print(f" - Arquivo: {audio_file.name}")
        print(f" - Motor: {self.engine}")
        print(f" - Dispositivo: {self.device.upper()} ({self.compute_type})")
        print(f" - Modo (task): {task} ({'Tradução para inglês' if task == 'translate' else 'Transcrição original'})")
        print(f" - Idioma: {language or 'Detecção automática'}")
        if self.engine in ("faster-whisper", "faster", "ctranslate2"):
            print(f" - Filtro VAD (Voice Activity): {vad_filter}")
        print("=" * 60 + "\n")

        start_time = time.time()

        if self.engine in ("faster-whisper", "faster", "ctranslate2"):
            segments_gen, info = model.transcribe(
                str(audio_file),
                language=language,
                task=task,
                initial_prompt=initial_prompt,
                vad_filter=vad_filter,
                temperature=temperature,
                beam_size=5,
            )

            segments = []
            for seg in segments_gen:
                seg_text = seg.text.strip()
                start_fmt = format_timestamp(seg.start, "srt")
                end_fmt = format_timestamp(seg.end, "srt")
                if verbose and seg_text:
                    print(f"[{start_fmt} --> {end_fmt}] {seg_text}")

                segments.append({
                    "id": getattr(seg, "id", len(segments)),
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                })

            full_text = " ".join(s["text"].strip() for s in segments)
            detected_lang = info.language
            result = {
                "text": full_text,
                "segments": segments,
                "language": detected_lang,
            }

        else:
            # Motor original openai-whisper
            kwargs: dict[str, Any] = {
                "task": task,
                "verbose": verbose,
                "fp16": use_fp16,
                "temperature": temperature,
            }
            if language:
                kwargs["language"] = language
            if initial_prompt:
                kwargs["initial_prompt"] = initial_prompt

            result = model.transcribe(str(audio_file), **kwargs)
            detected_lang = result.get("language", language)

        elapsed = time.time() - start_time
        total_segments = len(result.get("segments", []))

        print("\n" + "=" * 60)
        print(f"Transcrição concluída com sucesso!")
        print(f" - Tempo total: {elapsed:.2f}s ({elapsed / 60:.2f} min)")
        print(f" - Idioma: {detected_lang}")
        print(f" - Segmentos gerados: {total_segments}")
        print("=" * 60)

        # Libera memória residual da GPU
        if self.device == "cuda":
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

        return result
