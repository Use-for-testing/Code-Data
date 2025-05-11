#!/bin/bash
# Script to set up the code extraction toolkit in a GitHub repository

echo "GitHub Code Extraction Toolkit Setup"
echo "==================================="

# Check if Git is installed
if ! command -v git &> /dev/null; then
    echo "Error: Git is not installed. Please install Git and try again."
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Warning: Python 3 not found. The toolkit requires Python 3.6+ to run."
    echo "Please install Python 3 before using the toolkit."
fi

# Determine target repository
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Usage: $0 [target_repository_path]"
    echo ""
    echo "If no target repository is specified, the script will set up the toolkit in the current directory."
    exit 0
fi

TARGET_REPO=${1:-.}
TOOLKIT_DIR="$TARGET_REPO/Scripts"

# Check if target is a git repository
if [ ! -d "$TARGET_REPO/.git" ]; then
    echo "Warning: $TARGET_REPO does not appear to be a Git repository."
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Setup cancelled."
        exit 1
    fi
fi

# Create Scripts directory if it doesn't exist
echo "Setting up toolkit in: $TOOLKIT_DIR"
mkdir -p "$TOOLKIT_DIR"

# Copy toolkit files to target repository
SCRIPT_DIR=$(dirname "$0")
cp -v "$SCRIPT_DIR/extract_code_samples.py" "$TOOLKIT_DIR/"
cp -v "$SCRIPT_DIR/analyze_code_dataset.py" "$TOOLKIT_DIR/"
cp -v "$SCRIPT_DIR/run_extraction_and_analysis.sh" "$TOOLKIT_DIR/"
cp -v "$SCRIPT_DIR/run_extraction_and_analysis.bat" "$TOOLKIT_DIR/"
cp -v "$SCRIPT_DIR/requirements.txt" "$TOOLKIT_DIR/"
cp -v "$SCRIPT_DIR/README.md" "$TOOLKIT_DIR/"
cp -v "$SCRIPT_DIR/example_workflow.md" "$TOOLKIT_DIR/"

# Make scripts executable
chmod +x "$TOOLKIT_DIR/run_extraction_and_analysis.sh"

# Install dependencies if requested
read -p "Install Python dependencies now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing dependencies..."
    pip install -r "$TOOLKIT_DIR/requirements.txt"
    echo "Dependencies installed."
fi

echo
echo "Setup complete! The toolkit is now installed in: $TOOLKIT_DIR"
echo
echo "Quick start:"
echo "1. cd $TARGET_REPO"
echo "2. ./Scripts/run_extraction_and_analysis.sh"
echo
echo "For more examples, see: $TOOLKIT_DIR/example_workflow.md"
