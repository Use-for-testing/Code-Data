@echo off
SETLOCAL EnableDelayedExpansion

echo GitHub Code Extraction and Analysis Suite
echo =========================================

REM Default values
SET REPO_PATH=.
SET OUTPUT_DIR=.\dataset
SET MIN_LINES=5
SET MAX_SAMPLES=1000
SET EXPORT_FORMAT=all

REM Parse command line arguments
:parse_args
IF "%~1"=="" GOTO end_parse_args
IF "%~1"=="--repo-path" (
    SET REPO_PATH=%~2
    SHIFT
    SHIFT
    GOTO parse_args
)
IF "%~1"=="--output" (
    SET OUTPUT_DIR=%~2
    SHIFT
    SHIFT
    GOTO parse_args
)
IF "%~1"=="--min-lines" (
    SET MIN_LINES=%~2
    SHIFT
    SHIFT
    GOTO parse_args
)
IF "%~1"=="--max-samples" (
    SET MAX_SAMPLES=%~2
    SHIFT
    SHIFT
    GOTO parse_args
)
IF "%~1"=="--export-format" (
    SET EXPORT_FORMAT=%~2
    SHIFT
    SHIFT
    GOTO parse_args
)
ECHO Unknown argument: %~1
EXIT /B 1

:end_parse_args

REM Create the output directory if it doesn't exist
IF NOT EXIST "%OUTPUT_DIR%" MKDIR "%OUTPUT_DIR%"

echo.
echo Step 1: Installing required dependencies
echo ---------------------------------------
WHERE pip >nul 2>nul
IF %ERRORLEVEL% EQU 0 (
    pip install -r "%~dp0requirements.txt"
) ELSE (
    echo Warning: pip not found. Skipping dependency installation.
    echo Please manually install the required packages from requirements.txt
)

echo.
echo Step 2: Extracting code samples from repository
echo ---------------------------------------------
python "%~dp0extract_code_samples.py" ^
  --repo-path "%REPO_PATH%" ^
  --output "%OUTPUT_DIR%" ^
  --min-lines "%MIN_LINES%" ^
  --max-samples "%MAX_SAMPLES%" ^
  --export-format "%EXPORT_FORMAT%"

IF %ERRORLEVEL% NEQ 0 (
    echo Error during extraction. Exiting.
    EXIT /B %ERRORLEVEL%
)

echo.
echo Step 3: Analyzing extracted code dataset
echo --------------------------------------
python "%~dp0analyze_code_dataset.py" ^
  --dataset "%OUTPUT_DIR%" ^
  --output "%OUTPUT_DIR%\analysis" ^
  --format "json"

IF %ERRORLEVEL% NEQ 0 (
    echo Warning: Analysis completed with errors.
)

echo.
echo Process complete! Results are available in: %OUTPUT_DIR%
echo   - Code samples: %OUTPUT_DIR%\code_samples_dataset.json
echo   - Analysis: %OUTPUT_DIR%\analysis\code_analysis_report.html
echo.
echo To view the analysis report, open the HTML file in your browser:
echo   %OUTPUT_DIR%\analysis\code_analysis_report.html

ENDLOCAL
