@echo off
chcp 65001 >nul
cd /d %~dp0
echo ================================
echo   企业知识库 AI 助手 启动中...
echo   启动后请打开 http://127.0.0.1:8000
echo   停止服务：按 Ctrl+C
echo ================================
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" main.py
) else (
  python main.py
)
pause
