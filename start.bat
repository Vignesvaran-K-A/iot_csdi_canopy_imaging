@echo off
title Canopy Vision - Automated Image Processing Portal
echo ========================================================
echo   Launching Canopy Vision Automated Image Pipeline Portal
echo ========================================================
cd /d "%~dp0automated_image_processing_pipeline"
python run_portal.py
pause
