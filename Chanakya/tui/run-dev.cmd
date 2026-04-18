@echo off
setlocal
cd /d "%~dp0"
node --import tsx src\index.ts
