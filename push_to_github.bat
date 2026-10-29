@echo off
title Push TrueCost to GitHub
echo ========================================================
echo   Pushing TrueCost to GitHub
echo ========================================================
echo.
echo Pushing to https://github.com/Kartikeya-Rana-2004/TrueCost.git ...
echo.
git add -A
git commit -m "feat: add interactive token predictor live dashboard, github pages workflow, and results summary"
git push -u origin main
echo.
if %ERRORLEVEL% EQU 0 (
    echo ========================================================
    echo [SUCCESS] Pushed to GitHub successfully!
    echo.
    echo Next step: Enable GitHub Pages in your repo settings:
    echo 1. Go to: https://github.com/Kartikeya-Rana-2004/TrueCost/settings/pages
    echo 2. Under 'Build and deployment', choose 'GitHub Actions'
    echo    (The included workflow will deploy automatically!)
    echo.
    echo Your live dashboard will be online at:
    echo https://kartikeya-rana-2004.github.io/TrueCost/
    echo ========================================================
) else (
    echo [ERROR] Push failed.
)
echo.
pause
