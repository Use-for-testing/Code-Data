# Code Data

This repository contains a script for downloading and organizing programming language code samples.

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
   pip install huggingface_hub tqdm requests
   ```

2. Run the script:
   ```
   python Data.py
   ```

3. The script will create a `code_by_language` directory with subdirectories for each language, containing code samples.

### Features

- Downloads sample code from repositories or creates sample files
- Organizes files by programming language
- Detects language based on file extension
- Handles duplicate files
- Provides detailed logging

### Recent Fixes

The script has been updated to fix several issues:
1. Fixed the dataset download approach by using a fallback mechanism
2. Added proper language detection based on file extensions
3. Implemented sample file creation when dataset download fails
4. Added better error handling and logging
5. Added a .gitignore file to exclude generated files and directories
6. Improved code organization with separate functions for different tasks