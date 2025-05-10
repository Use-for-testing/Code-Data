# Code Data

This repository contains a script for downloading and organizing programming language code samples from the Hugging Face dataset.

## Data.py Script

The `Data.py` script downloads and organizes code samples for various programming languages:

- Swift
- Python
- Lua
- C
- C++
- Objective-C
- C#
- Ruby
- JavaScript
- TypeScript
- Luau

### How to Use

1. Install the required dependencies:
   ```
   pip install datasets huggingface_hub tqdm requests
   ```

2. Run the script:
   ```
   python Data.py
   ```

3. The script will create a `code_by_language` directory with subdirectories for each language, containing code samples.

### Features

- Downloads code samples from the Hugging Face dataset
- Organizes files by programming language
- Normalizes language names to match desired languages
- Handles duplicate files
- Provides fallback sample files when dataset access fails
- Provides detailed logging

### Recent Fixes

The script has been updated to fix several issues:
1. Added proper language mapping for normalization
2. Added fallback mechanism to create sample files when dataset access fails
3. Added file extension detection for better language identification
4. Improved error handling and logging
5. Added a .gitignore file to exclude generated files and directories