@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "PYTHON_EXE=python"

where python >nul 2>nul
if errorlevel 1 (
    where py >nul 2>nul
    if errorlevel 1 (
        echo 未找到 python 或 py，请先安装 Python 并加入 PATH。
        exit /b 1
    )
    set "PYTHON_EXE=py -3"
)

%PYTHON_EXE% "%SCRIPT_DIR%process_media.py" %*
exit /b %errorlevel%

