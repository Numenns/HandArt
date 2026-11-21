@echo off
cd /d "%~dp0.."
".venv\Scripts\python.exe" -m PyInstaller --noconfirm --distpath dist --workpath build packaging\HandArt.spec
