# 🎙️ Whisper Subtitle Generator (Aceleração Local por GPU)

Projeto adaptado a partir do Jupyter Notebook original (`legenda_whisper_v2.ipynb`) para um projeto Python local, modular, com suporte nativo à aceleração por hardware via GPU dedicada **NVIDIA GeForce RTX 5060 (8 GB)** e suporte a múltiplos formatos de legenda.

---

## 📋 Sumário
1. [Principais Mudanças em Relação ao Notebook](#-principais-mudanças-em-relação-ao-notebook)
2. [Diagnóstico do seu Ambiente](#-diagnóstico-do-seu-ambiente)
3. [Passo a Passo para Instalação e Execução](#-passo-a-passo-para-instalação-e-execução)
4. [Como Usar](#-como-usar)
   - [Via Linha de Comando (CLI)](#via-linha-de-comando-cli)
   - [Via Arrastar e Soltar (run.bat)](#via-arrastar-e-soltar-runbat)
   - [Exemplos Práticos](#exemplos-práticos)
5. [Instalações Complementares Necessárias](#-instalações-complementares-necessárias)
6. [Sugestões de Adequações e Próximos Passos](#-sugestões-de-adequações-e-próximos-passos)
7. [Checklist do Projeto](#-checklist-do-projeto)

---

## 🔄 Principais Mudanças em Relação ao Notebook

| Aspecto | No Google Colab (Notebook) | No Projeto Local (`src/`) |
| :--- | :--- | :--- |
| **Execução** | Preso a células do navegador e Google Drive | Script CLI autônomo e executador `.bat` |
| **GPU** | T4 instável na nuvem com timeout | **NVIDIA RTX 5060 8GB** local sem limite de tempo |
| **Caminhos** | Hardcoded (`/content/drive/...`) | Dinâmico: aceita qualquer arquivo local via argumento ou drag-and-drop |
| **Codificação** | `open()` padrão (podia corromper caracteres no Windows) | `encoding="utf-8"` explícito para japonês, português, emojis, etc. |
| **Formatos** | Apenas `.srt` | `.srt`, `.vtt` (WebVTT), `.txt` (texto puro) e `.json` completo |
| **Modo de Tarefa** | `task="translate"` fixo (traduzia tudo para inglês) | Configurável: `transcribe` (idioma original) ou `translate` (inglês) |
| **Modelos** | Fixo em `medium` | Opções: `tiny`, `base`, `small`, `medium`, `turbo`, `large-v3` |

---

## 💻 Diagnóstico do seu Ambiente

- **GPU**: NVIDIA GeForce RTX 5060 (8151 MiB VRAM) detectada via `nvidia-smi`.
- **Driver NVIDIA**: Versão 596.21 com suporte a CUDA 13.2 / 12.6+.
- **FFmpeg**: **Já instalado** no sistema (`ffmpeg version 8.0.1` disponível no PATH).
- **Python**: 3.14.2 disponível no ambiente virtual `.venv`.

---

## 🚀 Passo a Passo para Instalação e Execução

### Passo 1: Abrir o terminal no diretório do projeto
No PowerShell ou Prompt de Comando, certifique-se de estar na pasta do projeto:
```powershell
cd F:\Projetos\Whisper
```

### Passo 2: Ativar o ambiente virtual `.venv`
```powershell
.\.venv\Scripts\Activate.ps1
```
*(Se estiver no CMD tradicional, use: `.\.venv\Scripts\activate.bat`)*

### Passo 3: Instalar o PyTorch com suporte a CUDA
> [!IMPORTANT]
> A sua placa de vídeo (RTX 5060) necessita dos binários do PyTorch compilados com suporte a CUDA. Execute este comando específico:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu132
```

### Passo 4: Instalar as demais dependências do projeto
```powershell
pip install -r requirements.txt
```

### Passo 5: Confirmar a detecção da GPU
Execute o verificador rápido de hardware:
```powershell
python main.py --check-gpu
```
*Deverá exibir:*
```text
=== Diagnóstico de Hardware ===
Status GPU: Disponível via CUDA
Dispositivo: NVIDIA GeForce RTX 5060
VRAM Total: 7.96 GB
CUDA Runtime: 12.6
================================
```

---

## 🎯 Como Usar

### Via Linha de Comando (CLI)

#### 1. Transcrição Simples (Mantendo o idioma falado)
Gera o arquivo `.srt` na mesma pasta do arquivo original:
```powershell
python main.py "C:\Caminho\Para\SeuVideo.mp4"
```

#### 2. Tradução para o Inglês (Comportamento do notebook original)
Se o áudio estiver em japonês, português, coreano, etc., e você quiser a legenda traduzida para o inglês:
```powershell
python main.py "video.mp4" --task translate
```

#### 3. Especificar o Idioma
O Whisper detecta o idioma automaticamente, mas forçar o idioma melhora a velocidade e precisão inicial:
```powershell
python main.py "video.mp4" --language ja
```
*(Códigos comuns: `pt` = Português, `ja` = Japonês, `en` = Inglês, `es` = Espanhol, `ko` = Coreano)*

#### 4. Gerar múltiplos formatos de legenda
Gera `.srt`, `.vtt`, `.txt` e `.json` simultaneamente:
```powershell
python main.py "video.mp4" --format all
```
Ou apenas alguns:
```powershell
python main.py "video.mp4" --format srt,vtt,txt
```

#### 5. Escolher o Modelo do Whisper
A sua RTX 5060 com 8GB de VRAM roda com folga os seguintes modelos:
- `--model medium` *(Padrão)*: Excelente equilíbrio de qualidade e velocidade (~5 GB VRAM).
- `--model turbo`: Modelo mais recente otimizado para velocidade extrema sem perda notável de precisão (~6 GB VRAM).
- `--model large-v3`: Maior precisão possível (~7.5 GB VRAM).
- `--model small` ou `--model base`: Ultraleves e rápidos.

Exemplo:
```powershell
python main.py "video.mp4" --model turbo
```

#### 6. Especificar pasta de saída
```powershell
python main.py "video.mp4" --output-dir "D:\Legendas"
```

---

### Via Arrastar e Soltar (run.bat)

Você não precisa abrir o terminal sempre que quiser legendar um vídeo!
1. Abra a pasta do projeto no **Explorador de Arquivos do Windows**.
2. **Arraste e solte** o seu arquivo de vídeo ou áudio diretamente em cima do arquivo `run.bat`.
3. O script ativará a `.venv` e processará o arquivo na hora com a GPU.

---

## 🛠️ Instalações Complementares Necessárias

1. **FFmpeg**:
   - **Status atual**: ✅ **Já instalado e pronto** na sua máquina (`ffmpeg version 8.0.1`).
   - O Whisper utiliza o FFmpeg nos bastidores para extrair faixas de áudio de qualquer formato de vídeo.
2. **Drivers NVIDIA**:
   - **Status atual**: ✅ **Já instalado e atualizado** (Versão 596.21).
3. **PyTorch CUDA**:
   - Necessário rodar o comando do Passo 3 dentro da `.venv` para que o Python acesse a GPU em vez de usar a CPU.

---

## 💡 Sugestões de Adequações e Próximos Passos

Aqui estão melhorias e expansões recomendadas que podem ser adicionadas ao projeto:

1. **Migração opcional para `faster-whisper` (CTranslate2)**:
   - A biblioteca `faster-whisper` reescreve a inferência em C++ com suporte a quantização (FP16/INT8).
   - **Vantagem**: Reduz o consumo de VRAM pela metade e transcreve até **4x mais rápido** que o `openai-whisper` original.
2. **Processamento em Lote (Batch Processing)**:
   - Permitir passar uma pasta inteira de vídeos (ex: `python main.py --batch "C:\MeusVideos"`) para legendar todos em fila de processamento.
3. **Interface Gráfica Local (WebUI ou GUI Desktop)**:
   - Criar uma interface simples usando **Gradio** ou **CustomTkinter** com barra de progresso visual, player de pré-visualização e botões de download.
4. **Diarização de Locutores (Identificação de Falantes)**:
   - Integrar com `pyannote.audio` para identificar quem está falando em cada segmento (ex: `[Locutor 1]`, `[Locutor 2]`).
5. **Divisão Inteligente de Caracteres por Linha**:
   - Padronizar quebras de linha automáticas para players de vídeo (máximo de 37 a 42 caracteres por linha, padrão Netflix/YouTube) para evitar que trechos longos fiquem visualmente poluídos na tela.
6. **Word-Level Timestamps**:
   - Gerar legendas animadas palavra por palavra (estilo shorts do TikTok / Instagram / CapCut).

---

## ✅ Checklist do Projeto

- [x] Extração e adaptação da lógica do notebook `legenda_whisper_v2.ipynb`.
- [x] Estrutura modular (`src/transcriber.py`, `src/utils.py`, `src/cli.py`, `main.py`).
- [x] Detecção automática da GPU NVIDIA GeForce RTX 5060 com fallback seguro.
- [x] Codificação UTF-8 explícita para evitar problemas de acentuação e caracteres internacionais no Windows.
- [x] Suporte a múltiplos formatos de legenda (`.srt`, `.vtt`, `.txt`, `.json`).
- [x] Suporte aos modos `transcribe` (original) e `translate` (inglês).
- [x] `requirements.txt` estruturado.
- [x] Script `run.bat` para facilidade de uso com arrastar-e-soltar.
- [x] Testes unitários automatizados em `tests/test_utils.py` com 100% de aprovação.
- [x] Regras de metadados do Antigravity configuradas em `GEMINI.md` e `.antigravity/rules.md`.
