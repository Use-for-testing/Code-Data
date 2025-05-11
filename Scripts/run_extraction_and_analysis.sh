#!/bin/bash

# Script to run both the extraction and analysis tools
# Usage: ./run_extraction_and_analysis.sh [--repo-path PATH] [--output OUTPUT_DIR]

echo "GitHub Code Extraction and Analysis Suite"
echo "========================================="

# Default values
REPO_PATH="."
OUTPUT_DIR="./dataset"
MIN_LINES=5
MAX_SAMPLES=1000
EXPORT_FORMAT="all"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --repo-path)
      REPO_PATH="$2"
      shift 2
      ;;
    --output)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --min-lines)
      MIN_LINES="$2"
      shift 2
      ;;
    --max-samples)
      MAX_SAMPLES="$2"
      shift 2
      ;;
    --export-format)
      EXPORT_FORMAT="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo
echo "Step 1: Installing required dependencies"
echo "---------------------------------------"
if command -v pip &> /dev/null; then
  pip install -r "$(dirname "$0")/requirements.txt"
else
  echo "Warning: pip not found. Skipping dependency installation."
  echo "Please manually install the required packages from requirements.txt"
fi

echo
echo "Step 2: Extracting code samples from repository"
echo "---------------------------------------------"
python "$(dirname "$0")/extract_code_samples.py" \
  --repo-path "$REPO_PATH" \
  --output "$OUTPUT_DIR" \
  --min-lines "$MIN_LINES" \
  --max-samples "$MAX_SAMPLES" \
  --export-format "$EXPORT_FORMAT"

EXTRACT_STATUS=$?
if [ $EXTRACT_STATUS -ne 0 ]; then
  echo "Error during extraction. Exiting."
  exit $EXTRACT_STATUS
fi

echo
echo "Step 3: Analyzing extracted code dataset"
echo "--------------------------------------"
python "$(dirname "$0")/analyze_code_dataset.py" \
  --dataset "$OUTPUT_DIR" \
  --output "$OUTPUT_DIR/analysis" \
  --format "json"

ANALYSIS_STATUS=$?
if [ $ANALYSIS_STATUS -ne 0 ]; then
  echo "Warning: Analysis completed with errors."
fi

echo
echo "Process complete! Results are available in: $OUTPUT_DIR"
echo "  - Code samples: $OUTPUT_DIR/code_samples_dataset.json"
echo "  - Analysis: $OUTPUT_DIR/analysis/code_analysis_report.html"
echo
echo "To view the analysis report, open the HTML file in your browser:"
echo "  $OUTPUT_DIR/analysis/code_analysis_report.html"
