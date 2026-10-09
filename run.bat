@echo off
setlocal enableextensions
chcp 65001 >nul
cd /d "%~dp0"

echo ==================================================
echo   Real-time Hand Joint Angle Annotator
echo   MediaPipe HandLandmarker
echo ==================================================
echo.

REM ============================================================
REM  Find a Python that actually has the required packages.
REM
REM  Two traps this handles:
REM   1) conda's python.exe may not be on PATH at all
REM   2) "where python" often hits the Microsoft Store stub
REM      (%LOCALAPPDATA%\Microsoft\WindowsApps\python.exe)
REM      which is a 0-byte alias that opens the Store.
REM
REM  Strategy: try every candidate, accept the FIRST one that
REM            can actually import cv2 + mediapipe + numpy + PIL.
REM ============================================================
set "PY="

REM ---- 0) manual override (uncomment and set if auto-detect fails) ----
REM set "PY=D:\pkg\miniconda3\python.exe"
if defined PY goto gotpy

REM ---- 1) candidate: whatever is on PATH (store stub rejected in :probe) ----
for /f "delims=" %%p in ('where python 2^>nul') do (
    echo %%p| findstr /i /c:"WindowsApps" >nul || call :probe "%%p"
)
if defined PY goto gotpy

REM ---- 2) common conda / python install locations ----
for %%d in (
    "D:\pkg\miniconda3"
    "D:\pkg\anaconda3"
    "D:\miniconda3"
    "D:\anaconda3"
    "%USERPROFILE%\miniconda3"
    "%USERPROFILE%\anaconda3"
    "%LOCALAPPDATA%\miniconda3"
    "%LOCALAPPDATA%\Continuum\anaconda3"
    "C:\ProgramData\miniconda3"
    "C:\ProgramData\anaconda3"
    "C:\miniconda3"
    "C:\anaconda3"
        ) do call :probe "%%~d\python.exe"
if defined PY goto gotpy

REM ---- 3) plain Python installs ----
for %%d in (
    "%LOCALAPPDATA%\Programs\Python\Python313"
    "%LOCALAPPDATA%\Programs\Python\Python312"
    "%LOCALAPPDATA%\Programs\Python\Python311"
    "%LOCALAPPDATA%\Programs\Python\Python310"
    "C:\Python313"  "C:\Python312"  "C:\Python311"  "C:\Python310"
        ) do call :probe "%%~d\python.exe"
if defined PY goto gotpy

REM ---- 4) conda info --base ----
for /f "delims=" %%p in ('conda info --base 2^>nul') do call :probe "%%p\python.exe"
if defined PY goto gotpy

REM ---- 5) py launcher ----
for /f "delims=" %%p in ('where py 2^>nul') do call :probe "%%p"
if defined PY goto gotpy

goto nopython


REM ============================================================ found
:gotpy
echo   Python   : %PY%
"%PY%" -c "import sys;print('   version  :', sys.version.split()[0])"
echo   packages : OK  (cv2 / mediapipe / numpy / PIL)
echo.
echo   Keys:  q or ESC = quit        s = screenshot
echo          m = mirror toggle      space = pause
echo          r = record toggle
echo.
echo   Starting camera...
echo.

"%PY%" main.py --panel --auto-roi %*
set RC=%errorlevel%

if not "%RC%"=="0" goto runfail

echo.
echo [done] exited normally.
pause
exit /b 0


REM ============================================================ helpers
:probe
REM skip if already found / path missing / is the Store stub
if defined PY goto :eof
if not exist %1 goto :eof
echo %~1| findstr /i /c:"WindowsApps" >nul && goto :eof
REM must be a real interpreter WITH the packages
%~1 -c "import cv2, mediapipe, numpy, PIL" >nul 2>nul
if errorlevel 1 goto :eof
set "PY=%~1"
goto :eof


REM ============================================================ errors
:nopython
echo [!] Could not find a Python that has the required packages.
echo.
echo   Searched: PATH, common conda locations, conda info --base, py launcher
echo.
echo   What to do:
echo     1. Open a terminal and run:
echo          python -c "import cv2, mediapipe, numpy, PIL"
echo        If that fails, install the packages:
echo          python -m pip install -r requirements.txt
echo.
echo     2. If you know your python path, open run.bat in Notepad,
echo        find the line near the top:
echo            REM set "PY=D:\pkg\miniconda3\python.exe"
echo        remove "REM " and set it to your own path.
echo.
pause
exit /b 1

:runfail
echo.
echo [!] Program exited with code %RC%
echo.
echo  Common causes:
echo    1. Camera occupied by another app
echo       - close WeChat / QQ / browser / meeting software
echo    2. Wrong camera index
echo       - try:  run.bat --camera 1
echo    3. Windows camera permission
echo       - Settings - Privacy and security - Camera
echo       - allow desktop apps to access the camera
echo.
pause
exit /b %RC%
