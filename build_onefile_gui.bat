@echo off
setlocal
cd /d "%~dp0"

if not exist "app_icon.ico" (
  echo.
  echo ERROR: app_icon.ico not found in this folder.
  echo Put your custom Windows .ico file here and name it app_icon.ico.
  echo.
  pause
  exit /b 1
)

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
call venv\Scripts\activate.bat

python -m pip install -r requirements.txt
python -m pip install pyinstaller

rmdir /s /q build 2>nul
rmdir /s /q dist 2>nul

echo.
echo Building single EXE. This can take a while...
echo.

python -m PyInstaller ^
  --noconfirm ^
  --clean ^
  --onefile ^
  --windowed ^
  --name "GameMaster" ^
  --icon "app_icon.ico" ^
  --hidden-import main ^
  --hidden-import run ^
  --hidden-import config.settings ^
  --hidden-import gm.retriever ^
  --hidden-import gm.prompt_filter ^
  --hidden-import gm.selector ^
  --hidden-import gm.static_index ^
  --collect-submodules uvicorn ^
  --collect-submodules fastapi ^
  --collect-submodules starlette ^
  --collect-submodules pydantic ^
  --collect-submodules httpx ^
  --collect-submodules anyio ^
  gamemaster_gui.py

if errorlevel 1 (
  echo.
  echo Build failed.
  pause
  exit /b 1
)

echo.
echo Built: dist\GameMaster.exe
echo Put only this EXE wherever you want. On first launch it will create config\settings.json next to itself.
echo.
pause
