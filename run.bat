@echo off
chcp 65001 > nul
setlocal

:: Navega para a pasta do script
cd /d "%~dp0"

echo ======================================================================
echo           Whisper Subtitle Generator - Acelerado por GPU CUDA
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
    echo Arraste e solte um arquivo de video ou audio diretamente sobre este arquivo run.bat!
    echo.
    echo Ou informe o caminho do arquivo agora:
    set /p MEDIA_FILE="Digite ou arraste o arquivo aqui: "
    if "%MEDIA_FILE%"=="" (
        echo Nenhum arquivo fornecido. Encerrando.
        pause
        exit /b 0
    )
    "%PYTHON_EXE%" main.py "%MEDIA_FILE%"
) else (
    echo.
    echo Processando arquivo: %~1
    echo.
    "%PYTHON_EXE%" main.py "%~1"
)

echo.
echo ======================================================================
echo Processo finalizado!
pause
