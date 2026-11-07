@echo off
setlocal
cd /d "%~dp0"

if not exist "llm\Scripts\python.exe" (
    py -3.11 -m venv llm
    if errorlevel 1 (
        echo Could not create the llm environment. Install Python 3.11 and try again.
        exit /b 1
    )
)

"llm\Scripts\python.exe" -c "import dotenv, flask, openai, pypdf, werkzeug" >nul 2>nul
if errorlevel 1 (
    "llm\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Could not install project dependencies into the llm environment.
        exit /b 1
    )
)

"llm\Scripts\python.exe" app.py
exit /b %errorlevel%
