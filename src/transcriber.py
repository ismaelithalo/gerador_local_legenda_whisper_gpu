"""Módulo responsável pelo carregamento do Whisper e execução da transcrição com suporte a GPU CUDA."""

from __future__ import annotations
import sys
import time
from pathlib import Path
from typing import Any, Optional


def check_gpu_status() -> dict[str, Any]:
    """
    Verifica a disponibilidade de GPU dedicada via PyTorch CUDA.
    Retorna informações detalhadas sobre a GPU ou CPU.
    """
    try:
        import torch
    except ImportError:
        return {
            "available": False,
            "device": "cpu",
            "name": "PyTorch não instalado",
            "vram_gb": 0.0,
            "cuda_version": None,
        }

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
    else:
        return {
            "available": False,
            "device": "cpu",
            "name": "Processador (CPU)",
            "vram_gb": 0.0,
            "cuda_version": None,
        }


class WhisperTranscriber:
    """Gerencia o ciclo de vida do modelo Whisper e a transcrição acelerada por hardware."""

    def __init__(
        self,
        model_name: str = "medium",
        device: str = "auto",
        download_root: Optional[str | Path] = None,
    ):
        """
        Inicializa o transcritor.

        Args:
            model_name: tiny, base, small, medium, large-v3, turbo
            device: 'auto', 'cuda', ou 'cpu'
            download_root: Diretório onde o cache dos modelos será salvo
        """
        self.model_name = model_name
        self.download_root = str(download_root) if download_root else None
        self.gpu_info = check_gpu_status()
        self.device = self._resolve_device(device)
        self.model = None

    def _resolve_device(self, requested_device: str) -> str:
        req = requested_device.lower().strip()
        if req == "auto":
            if self.gpu_info["available"]:
                print(f"[Hardware] GPU dedicada detectada: {self.gpu_info['name']} ({self.gpu_info['vram_gb']} GB VRAM)")
                return "cuda"
            else:
                print("[Hardware] Nenhuma GPU CUDA detectada. Utilizando CPU.")
                print("           Dica: Para habilitar aceleração por hardware, instale o PyTorch com suporte a CUDA.")
                return "cpu"
        elif req == "cuda":
            if not self.gpu_info["available"]:
                print("[Aviso] GPU CUDA solicitada, mas não está disponível no PyTorch atual. Revertendo para CPU.")
                return "cpu"
            return "cuda"
        elif req == "cpu":
            return "cpu"
        else:
            raise ValueError(f"Dispositivo desconhecido: {requested_device}. Use 'auto', 'cuda' ou 'cpu'.")

    def load_model(self):
        """Carrega os pesos do modelo Whisper na memória (ou GPU VRAM)."""
        if self.model is not None:
            return self.model

        try:
            import whisper
        except ImportError:
            raise ImportError(
                "O pacote 'openai-whisper' não está instalado.\n"
                "Execute: pip install openai-whisper"
            )

        print(f"\n[Whisper] Carregando modelo '{self.model_name}' em [{self.device.upper()}]...")
        start_time = time.time()
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
        temperature: float = 0.0,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """
        Executa a transcrição do arquivo de áudio ou vídeo.

        Args:
            audio_path: Caminho para o arquivo de mídia.
            language: Código do idioma (ex: 'pt', 'ja', 'en') ou None para detecção automática.
            task: 'transcribe' (mantém o idioma original) ou 'translate' (traduz para o inglês).
            initial_prompt: Dica de texto ou contexto prévio para guiar o Whisper.
            temperature: Temperatura de amostragem (padrão: 0.0 para maior precisão).
            verbose: Se True, exibe o progresso dos trechos em tempo real.

        Returns:
            Dicionário com o resultado completo ('text', 'segments', 'language').
        """
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {audio_file.resolve()}")

        model = self.load_model()

        # FP16 só é suportado com aceleração em GPU CUDA
        use_fp16 = (self.device == "cuda")

        print(f"\n" + "=" * 60)
        print(f"Iniciando transcrição:")
        print(f" - Arquivo: {audio_file.name}")
        print(f" - Dispositivo: {self.device.upper()}")
        print(f" - Modo (task): {task} ({'Tradução para inglês' if task == 'translate' else 'Transcrição no idioma original'})")
        print(f" - Idioma: {language or 'Detecção automática'}")
        print(f" - FP16 ativado: {use_fp16}")
        print("=" * 60 + "\n")

        start_time = time.time()

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

        elapsed = time.time() - start_time
        detected_lang = result.get("language", language)
        total_segments = len(result.get("segments", []))

        print("\n" + "=" * 60)
        print(f"Transcrição concluída com sucesso!")
        print(f" - Tempo total: {elapsed:.2f}s ({elapsed / 60:.2f} min)")
        print(f" - Idioma detectado/usado: {detected_lang}")
        print(f" - Segmentos gerados: {total_segments}")
        print("=" * 60)

        # Libera memória residual da GPU se aplicável
        if self.device == "cuda":
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

        return result
