# 🎙️ Whisper Subtitle Generator (Aceleração Local por GPU)

Projeto local em Python para transcrição e geração de legendas a partir de áudios e vídeos, adaptado do notebook original (`legenda_whisper_v2.ipynb`). Acelerado pela GPU dedicada **NVIDIA GeForce RTX 5060 (8 GB)** com suporte ao motor ultrarrápido **faster-whisper (CTranslate2)** e processamento em lote.

---

## 📋 Sumário
1. [Principais Mudanças em Relação ao Notebook](#-principais-mudanças-em-relação-ao-notebook)
2. [Diagnóstico do seu Ambiente](#-diagnóstico-do-seu-ambiente)
3. [Passo a Passo para Instalação e Execução](#-passo-a-passo-para-instalação-e-execução)
4. [Como Usar](#-como-usar)
   - [Via Linha de Comando (CLI)](#via-linha-de-comando-cli)
   - [Processamento em Lote (Batch)](#processamento-em-lote-batch)
   - [Via Arrastar e Soltar (run.bat)](#via-arrastar-e-soltar-runbat)
   - [Exemplos Práticos e Motores](#exemplos-práticos-e-motores)
5. [Instalações Complementares Necessárias](#-instalações-complementares-necessárias)
6. [Sugestões de Adequações e Próximos Passos](#-sugestões-de-adequações-e-próximos-passos)
7. [Checklist do Projeto](#-checklist-do-projeto)

---

## 🔄 Principais Mudanças em Relação ao Notebook

| Aspecto | No Google Colab (Notebook) | No Projeto Local (`src/`) |
| :--- | :--- | :--- |
| **Execução** | Preso a células do navegador e Google Drive | Script CLI autônomo e executador `.bat` |
| **GPU** | T4 instável na nuvem com timeout | **NVIDIA RTX 5060 8GB** local sem limite de tempo |
| **Motores** | Apenas OpenAI Whisper original | **faster-whisper (padrão, 4x mais rápido)** ou `openai-whisper` |
| **Lote (Batch)** | Manual, um arquivo por vez alterando código | Flag `--batch "pasta/"` ou arrastar pasta inteira |
| **Caminhos** | Hardcoded (`/content/drive/...`) | Dinâmico: aceita arquivos/pastas via argumento ou drag-and-drop |
| **Codificação** | `open()` padrão (podia corromper caracteres no Windows) | `encoding="utf-8"` explícito para japonês, português, emojis, etc. |
| **Formatos** | Apenas `.srt` | `.srt`, `.vtt` (WebVTT), `.txt` (texto puro) e `.json` completo |
| **Modo de Tarefa** | `task="translate"` fixo (traduzia tudo para inglês) | Configurável: `transcribe` (idioma original) ou `translate` (inglês) |
| **Modelos** | Fixo em `medium` | Opções: `tiny`, `base`, `small`, `medium`, `turbo`, `large-v3` |

---

## 💻 Diagnóstico do seu Ambiente

- **GPU**: NVIDIA GeForce RTX 5060 (8151 MiB VRAM) detectada via `nvidia-smi`.
- **Driver NVIDIA**: Versão 596.21 com suporte a CUDA 13.2 / 12.6+.
- **FFmpeg**: **Já instalado** no sistema (`ffmpeg version 8.0.1` disponível no PATH).
- **Python**: 3.14.2 configurado no ambiente virtual `.venv`.
- **PyTorch**: Compilação nativa com CUDA 13.2 instalada (`torch-2.14.1+cu132`).
- **CTranslate2 & cuBLAS/cuDNN**: Bibliotecas CUDA 12 instaladas e integradas para o faster-whisper.

---

## 🚀 Passo a Passo para Instalação e Execução

### Passo 1: Abrir o terminal no diretório do projeto
No PowerShell ou Prompt de Comando:
```powershell
cd F:\Projetos\Whisper
```

### Passo 2: Ativar o ambiente virtual `.venv`
```powershell
.\.venv\Scripts\Activate.ps1
```
*(Se estiver no CMD tradicional, use: `.\.venv\Scripts\activate.bat`)*

### Passo 3: Instalar o PyTorch com suporte a CUDA (Se for reinstalar do zero)
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu132
```

### Passo 4: Instalar as dependências do projeto
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
CUDA Runtime: 13.2
================================
```

---

## 🎯 Como Usar

### Via Linha de Comando (CLI)

#### 1. Transcrição Simples de Arquivo Único
Gera o arquivo `.srt` na mesma pasta do arquivo original com o motor `faster-whisper`:
```powershell
python main.py "C:\Caminho\Para\SeuVideo.mp4"
```

#### 2. Processamento em Lote (Batch)
Transcreve todos os vídeos e áudios de uma pasta automaticamente:
```powershell
python main.py --batch "C:\MinhaPastaComVideos"
```
Ou passando diretamente a pasta como argumento:
```powershell
python main.py "C:\MinhaPastaComVideos"
```
*Dica:* Adicione `--recursive` para incluir arquivos em todas as subpastas:
```powershell
python main.py --batch "C:\MinhaPastaComVideos" --recursive
```

#### 3. Escolha do Motor (`faster-whisper` vs `openai`)
O padrão é o **faster-whisper**, mas você pode alternar facilmente:
```powershell
# Usando o motor padrão faster-whisper (ultrarrápido, CTranslate2):
python main.py "video.mp4" --engine faster-whisper

# Usando o motor original do OpenAI Whisper:
python main.py "video.mp4" --engine openai
```

#### 4. Tradução para o Inglês (Comportamento do notebook original)
Se o áudio estiver em japonês, português, coreano, etc., e você quiser a legenda traduzida para o inglês:
```powershell
python main.py "video.mp4" --task translate
```

#### 5. Especificar o Idioma
O Whisper detecta o idioma automaticamente, mas forçar o idioma melhora a velocidade e precisão inicial:
```powershell
python main.py "video.mp4" --language ja
```
*(Códigos comuns: `pt` = Português, `ja` = Japonês, `en` = Inglês, `es` = Espanhol, `ko` = Coreano)*

#### 6. Gerar múltiplos formatos de legenda
Gera `.srt`, `.vtt`, `.txt` e `.json` simultaneamente:
```powershell
python main.py "video.mp4" --format all
```
Ou apenas alguns:
```powershell
python main.py "video.mp4" --format srt,vtt,txt
```

#### 7. Escolher o Modelo do Whisper
A sua RTX 5060 com 8GB de VRAM roda com folga os seguintes modelos:
- `--model medium` *(Padrão)*: Excelente equilíbrio de qualidade e velocidade (~2.5 GB no faster-whisper).
- `--model turbo`: Modelo mais recente otimizado para velocidade sem perda notável de precisão (~3 GB VRAM).
- `--model large-v3`: Maior precisão possível (~4-5 GB VRAM no faster-whisper).
- `--model small` ou `--model base`: Ultraleves e rápidos.

Exemplo:
```powershell
python main.py "video.mp4" --model turbo
```

#### 8. Especificar pasta de saída
```powershell
python main.py "video.mp4" --output-dir "D:\Legendas"
```

---

### Via Arrastar e Soltar (run.bat)

Você não precisa abrir o terminal sempre que quiser legendar!
1. Abra a pasta do projeto no **Explorador de Arquivos do Windows**.
2. **Arraste e solte** o seu arquivo de vídeo, áudio **OU UMA PASTA INTEIRA** diretamente em cima do arquivo `run.bat`.
3. Se você soltar um arquivo, ele processa o arquivo. Se você soltar uma pasta, ele ativa o modo em lote e legenda todos os vídeos da pasta em fila!

---

## 🛠️ Instalações Complementares Necessárias

1. **FFmpeg**:
   - **Status atual**: ✅ **Já instalado e pronto** na sua máquina (`ffmpeg version 8.0.1`).
2. **Drivers NVIDIA**:
   - **Status atual**: ✅ **Já instalado e atualizado** (Versão 596.21).
3. **PyTorch CUDA & CTranslate2**:
   - **Status atual**: ✅ **Já instalado e configurado** no ambiente `.venv`.

---

## 💡 Sugestões de Adequações e Próximos Passos

1. **Interface Gráfica Local (WebUI ou GUI Desktop)**:
   - Criar uma interface simples usando **Gradio** ou **CustomTkinter** com barra de progresso visual, player de pré-visualização e botões de download.
2. **Diarização de Locutores (Identificação de Falantes)**:
   - Integrar com `pyannote.audio` para identificar quem está falando em cada segmento (ex: `[Locutor 1]`, `[Locutor 2]`).
3. **Divisão Inteligente de Caracteres por Linha**:
   - Padronizar quebras de linha automáticas para players de vídeo (máximo de 37 a 42 caracteres por linha, padrão Netflix/YouTube) para evitar que trechos longos fiquem visualmente poluídos na tela.
4. **Word-Level Timestamps**:
   - Gerar legendas animadas palavra por palavra (estilo shorts do TikTok / Instagram / CapCut).

---

## ✅ Checklist do Projeto

- [x] Extração e adaptação da lógica do notebook `legenda_whisper_v2.ipynb`.
- [x] Estrutura modular (`src/transcriber.py`, `src/utils.py`, `src/cli.py`, `src/batch.py`, `main.py`).
- [x] Detecção automática da GPU NVIDIA GeForce RTX 5060 (CUDA).
- [x] Suporte ao motor `faster-whisper` (CTranslate2) com aceleração CUDA e quantização FP16.
- [x] Suporte contínuo ao motor original `openai-whisper` como alternativa selecionável.
- [x] Processamento em Lote com flag `--batch "pasta/"` e suporte a subpastas (`--recursive`).
- [x] Codificação UTF-8 explícita para evitar problemas de caracteres internacionais no Windows.
- [x] Suporte a múltiplos formatos de legenda (`.srt`, `.vtt`, `.txt`, `.json`).
- [x] Suporte aos modos `transcribe` (original) e `translate` (inglês).
- [x] `requirements.txt` estruturado com todas as bibliotecas.
- [x] Script `run.bat` aprimorado com suporte a arrastar-e-soltar para arquivos e pastas.
- [x] Testes unitários automatizados em `tests/` com 100% de aprovação.
- [x] Regras de metadados do Antigravity configuradas em `GEMINI.md` e `.antigravity/rules.md`.
