#!/usr/bin/env python3
"""
GitHub Code Sample Extractor

This script traverses a GitHub repository, extracts code samples from various
programming languages, and creates a structured dataset.

Usage:
    python extract_code_samples.py [--output OUTPUT_DIR] [--min-lines MIN_LINES] [--max-samples MAX_SAMPLES]

The script will:
1. Scan the current repository (or provided path)
2. Identify code files based on extensions
3. Extract code samples with metadata
4. Save the dataset in JSON format
"""

import os
import json
import argparse
import random
from pathlib import Path
from collections import defaultdict
import datetime
import shutil
import re

# Language detection based on file extensions
LANGUAGE_EXTENSIONS = {
    # Web development
    'js': 'JavaScript',
    'jsx': 'JavaScript (React)',
    'ts': 'TypeScript',
    'tsx': 'TypeScript (React)',
    'html': 'HTML',
    'css': 'CSS',
    'scss': 'SCSS',
    'less': 'LESS',
    
    # Mobile development
    'swift': 'Swift',
    'kt': 'Kotlin',
    'java': 'Java',
    'h': 'C/Objective-C Header',
    'm': 'Objective-C',
    'mm': 'Objective-C++',
    
    # General programming
    'py': 'Python',
    'rb': 'Ruby',
    'php': 'PHP',
    'go': 'Go',
    'rs': 'Rust',
    'c': 'C',
    'cpp': 'C++',
    'cc': 'C++',
    'cxx': 'C++',
    'cs': 'C#',
    'fs': 'F#',
    'pl': 'Perl',
    'sh': 'Shell',
    'bash': 'Bash',
    'lua': 'Lua',
    'r': 'R',
    'dart': 'Dart',
    'jl': 'Julia',
    'ex': 'Elixir',
    'exs': 'Elixir',
    'elm': 'Elm',
    'clj': 'Clojure',
    'scala': 'Scala',
    'hs': 'Haskell',
    'erl': 'Erlang',
    
    # Configuration and data
    'json': 'JSON',
    'yaml': 'YAML',
    'yml': 'YAML',
    'toml': 'TOML',
    'xml': 'XML',
    'sql': 'SQL',
    'graphql': 'GraphQL',
    'proto': 'Protocol Buffer',
    
    # Game development
    'gd': 'GDScript',
    'cs': 'C# (Unity)',
    'unity': 'Unity',
    'unrealscript': 'UnrealScript',
    
    # Other
    'md': 'Markdown',
    'rst': 'reStructuredText',
    'tex': 'LaTeX',
    'gradle': 'Gradle',
    'bat': 'Batch',
    'ps1': 'PowerShell',
    'vb': 'Visual Basic',
    'asm': 'Assembly',
    'vue': 'Vue',
    'svelte': 'Svelte',
}

# Directories to ignore during traversal
IGNORE_DIRS = {
    '.git', '.github', 'node_modules', 'venv', '.venv', '.env', 
    'env', '__pycache__', 'build', 'dist', 'target', 'out',
    '.idea', '.vscode', '.DS_Store', 'bin', 'obj', '.next',
    'coverage', '.coverage', 'tmp', 'temp', 'log', 'logs'
}

# Files to ignore 
IGNORE_FILES = {
    '.gitignore', '.gitattributes', '.gitmodules', '.npmrc', '.npmignore',
    'package-lock.json', 'yarn.lock', 'Pipfile.lock', 'Gemfile.lock',
    '.eslintrc', '.prettierrc', '.editorconfig', '.DS_Store', 'thumbs.db',
    'LICENSE', 'LICENSE.md', 'LICENSE.txt', 'NOTICE', 'CONTRIBUTORS',
    'CODEOWNERS', '.travis.yml', 'appveyor.yml', 'azure-pipelines.yml',
    'Dockerfile', 'docker-compose.yml', '.dockerignore'
}

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Extract code samples from a GitHub repository")
    parser.add_argument("--repo-path", type=str, default=".",
                        help="Path to the repository (default: current directory)")
    parser.add_argument("--output", type=str, default="./dataset",
                        help="Output directory for the dataset (default: ./dataset)")
    parser.add_argument("--min-lines", type=int, default=5,
                        help="Minimum number of lines for a code sample (default: 5)")
    parser.add_argument("--max-samples", type=int, default=1000,
                        help="Maximum number of samples per language (default: 1000)")
    parser.add_argument("--include-dirs", type=str, default="",
                        help="Comma-separated list of directories to include (default: all)")
    parser.add_argument("--include-langs", type=str, default="",
                        help="Comma-separated list of languages to include (default: all)")
    parser.add_argument("--exclude-dirs", type=str, default="",
                        help="Comma-separated list of additional directories to exclude")
    parser.add_argument("--exclude-langs", type=str, default="",
                        help="Comma-separated list of languages to exclude")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--export-format", type=str, choices=["json", "csv", "jsonl", "all"], 
                        default="json", help="Export format for the dataset")
    return parser.parse_args()

def is_binary_file(file_path, sample_size=8192):
    """Check if a file is binary by reading a sample and looking for null bytes."""
    try:
        with open(file_path, 'rb') as f:
            sample = f.read(sample_size)
            # If there's a null byte in the sample, it's likely binary
            if b'\x00' in sample:
                return True
            # Also check for non-text characters
            try:
                sample.decode('utf-8')
                return False
            except UnicodeDecodeError:
                return True
    except Exception:
        # If there was an error reading the file, consider it binary to be safe
        return True

def detect_language(file_path):
    """Detect programming language from file extension."""
    extension = file_path.split('.')[-1].lower() if '.' in file_path else ""
    return LANGUAGE_EXTENSIONS.get(extension, "Unknown")

def should_process_file(file_path, args, include_langs, exclude_langs):
    """Determine if a file should be processed based on criteria."""
    # Skip files in IGNORE_FILES
    if os.path.basename(file_path) in IGNORE_FILES:
        return False
    
    # Skip binary files
    if is_binary_file(file_path):
        return False
        
    # Check if the language is in the include list or exclude list
    language = detect_language(file_path)
    if language == "Unknown":
        return False
        
    if include_langs and language not in include_langs:
        return False
        
    if language in exclude_langs:
        return False
    
    return True

def extract_code_sample(file_path, min_lines=5):
    """Extract code from a file with metadata."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        lines = content.split('\n')
        if len(lines) < min_lines:
            return None
            
        language = detect_language(file_path)
        rel_path = os.path.relpath(file_path)
        
        # Try to extract class/function definition using regex
        code_type = "unknown"
        code_name = ""
        
        if language in ["Python", "Ruby", "JavaScript", "TypeScript"]:
            # Look for function or class definitions
            class_match = re.search(r'class\s+(\w+)', content)
            func_match = re.search(r'(def|function)\s+(\w+)', content)
            
            if class_match:
                code_type = "class"
                code_name = class_match.group(1)
            elif func_match:
                code_type = "function"
                code_name = func_match.group(2)
        
        return {
            "language": language,
            "file_path": rel_path,
            "content": content,
            "line_count": len(lines),
            "code_type": code_type,
            "code_name": code_name,
            "file_size_bytes": os.path.getsize(file_path),
            "extraction_timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def traverse_repository(args):
    """Traverse the repository and extract code samples."""
    repo_path = Path(args.repo_path).resolve()
    print(f"Scanning repository at: {repo_path}")
    
    # Process include/exclude lists
    include_dirs = set(args.include_dirs.split(',')) if args.include_dirs else set()
    exclude_dirs = set(IGNORE_DIRS)
    if args.exclude_dirs:
        exclude_dirs.update(args.exclude_dirs.split(','))
    
    include_langs = set(args.include_langs.split(',')) if args.include_langs else set()
    exclude_langs = set(args.exclude_langs.split(',')) if args.exclude_langs else set()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
    
    # Collect samples by language
    samples_by_language = defaultdict(list)
    
    # Walk through the repository
    for root, dirs, files in os.walk(repo_path):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        # If include_dirs is provided, only include those directories
        if include_dirs and not any(d in root for d in include_dirs):
            continue
            
        for file in files:
            file_path = os.path.join(root, file)
            
            if should_process_file(file_path, args, include_langs, exclude_langs):
                sample = extract_code_sample(file_path, args.min_lines)
                if sample:
                    language = sample["language"]
                    # Only add if we haven't reached the max samples for this language
                    if len(samples_by_language[language]) < args.max_samples:
                        samples_by_language[language].append(sample)
    
    # Create a flat list of all samples
    all_samples = []
    for language, samples in samples_by_language.items():
        all_samples.extend(samples)
    
    # Shuffle the samples to ensure diversity
    random.shuffle(all_samples)
    
    return all_samples, samples_by_language

def export_dataset(samples, samples_by_language, args):
    """Export the collected samples to a dataset."""
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create metadata about the dataset
    metadata = {
        "dataset_name": "GitHub Code Samples",
        "description": "A dataset of code samples extracted from a GitHub repository",
        "creation_date": datetime.datetime.now().isoformat(),
        "total_samples": len(samples),
        "languages": {lang: len(lang_samples) for lang, lang_samples in samples_by_language.items()},
        "extraction_parameters": {
            "min_lines": args.min_lines,
            "max_samples_per_language": args.max_samples,
            "included_directories": args.include_dirs.split(',') if args.include_dirs else "all",
            "excluded_directories": list(IGNORE_DIRS) + (args.exclude_dirs.split(',') if args.exclude_dirs else []),
            "included_languages": args.include_langs.split(',') if args.include_langs else "all",
            "excluded_languages": args.exclude_langs.split(',') if args.exclude_langs else [],
        }
    }
    
    # Export in the requested format(s)
    if args.export_format in ["json", "all"]:
        with open(output_dir / "code_samples_dataset.json", 'w', encoding='utf-8') as f:
            json.dump({"metadata": metadata, "samples": samples}, f, indent=2)
        
        # Also save a separate file for each language
        lang_dir = output_dir / "by_language"
        lang_dir.mkdir(exist_ok=True)
        
        for language, lang_samples in samples_by_language.items():
            safe_lang_name = language.replace('/', '_').replace('#', 'Sharp').replace('+', 'Plus')
            with open(lang_dir / f"{safe_lang_name}.json", 'w', encoding='utf-8') as f:
                json.dump({"language": language, "samples": lang_samples}, f, indent=2)
    
    if args.export_format in ["csv", "all"]:
        import csv
        
        # Create a flattened version of the data for CSV export
        with open(output_dir / "code_samples_dataset.csv", 'w', newline='', encoding='utf-8') as f:
            fieldnames = ["language", "file_path", "line_count", "code_type", "code_name", "file_size_bytes", "extraction_timestamp"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            for sample in samples:
                row = {field: sample[field] for field in fieldnames}
                writer.writerow(row)
    
    if args.export_format in ["jsonl", "all"]:
        with open(output_dir / "code_samples_dataset.jsonl", 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')
    
    # Always save metadata separately
    with open(output_dir / "dataset_metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nDataset exported to: {output_dir}")
    print(f"Total samples: {len(samples)}")
    print("Samples by language:")
    for language, count in sorted(metadata["languages"].items(), key=lambda x: x[1], reverse=True):
        print(f"  {language}: {count}")

def main():
    """Main function to run the code extraction."""
    args = parse_arguments()
    
    print("Starting GitHub Code Sample Extractor")
    print("-------------------------------------")
    
    # Traverse the repository and collect samples
    print("\nScanning repository for code samples...")
    samples, samples_by_language = traverse_repository(args)
    
    if not samples:
        print("No code samples found matching the criteria.")
        return
    
    # Export the dataset
    print(f"\nFound {len(samples)} code samples across {len(samples_by_language)} languages.")
    export_dataset(samples, samples_by_language, args)
    
    print("\nExtraction complete!")

if __name__ == "__main__":
    main()
