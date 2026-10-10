@echo off
chcp 65001 > nul
setlocal

:: Navega para a pasta do script
cd /d "%~dp0"

echo ======================================================================
echo       Whisper Subtitle Generator (faster-whisper + GPU CUDA)
echo ======================================================================

set PYTHON_EXE=.venv\Scripts\python.exe

if not exist "%PYTHON_EXE%" (
    echo [ERRO] O ambiente virtual .venv nao foi encontrado!
    echo Certifique-se de que a pasta .venv existe.
    pause
    exit /b 1
)

if "%~1"=="" (
    echo.
    echo [USO DIRETO]
    echo Arraste e solte um arquivo ou pasta de videos/audios diretamente sobre este run.bat!
    echo.
    echo Ou informe o caminho do arquivo ou pasta agora:
    set /p MEDIA_PATH="Digite ou arraste o arquivo ou pasta aqui: "
    if "%MEDIA_PATH%"=="" (
        echo Nenhum caminho fornecido. Encerrando.
        pause
        exit /b 0
    )
    if exist "%MEDIA_PATH%\*" (
        echo.
        echo Detectada pasta! Iniciando processamento em lote...
        "%PYTHON_EXE%" main.py --batch "%MEDIA_PATH%"
    ) else (
        echo.
        echo Processando arquivo unico...
        "%PYTHON_EXE%" main.py "%MEDIA_PATH%"
    )
) else (
    if exist "%~1\*" (
        echo.
        echo Pasta detectada: %~1
        echo Iniciando processamento em lote...
        "%PYTHON_EXE%" main.py --batch "%~1"
    ) else (
        echo.
        echo Processando arquivo: %~1
        echo.
        "%PYTHON_EXE%" main.py "%~1"
    )
)

echo.
echo ======================================================================
echo Processo finalizado!
pause
