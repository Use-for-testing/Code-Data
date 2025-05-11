# GitHub Code Sample Extractor

This directory contains scripts to extract code samples from a GitHub repository and organize them into a structured dataset.

## Main Script

### `extract_code_samples.py`

This script traverses a GitHub repository, identifies code files based on their extensions, extracts code samples with metadata, and creates a dataset in various formats.

## Usage

```bash
python extract_code_samples.py [OPTIONS]
```

### Options

- `--repo-path PATH`: Path to the repository (default: current directory)
- `--output DIR`: Output directory for the dataset (default: ./dataset)
- `--min-lines N`: Minimum number of lines for a code sample (default: 5)
- `--max-samples N`: Maximum number of samples per language (default: 1000)
- `--include-dirs DIRS`: Comma-separated list of directories to include (default: all)
- `--include-langs LANGS`: Comma-separated list of languages to include (default: all)
- `--exclude-dirs DIRS`: Comma-separated list of additional directories to exclude
- `--exclude-langs LANGS`: Comma-separated list of languages to exclude
- `--seed N`: Random seed for reproducibility
- `--export-format FORMAT`: Export format for the dataset (json, csv, jsonl, or all, default: json)

## Examples

### Basic usage

```bash
python Scripts/extract_code_samples.py
```

This will scan the current repository and create a dataset in the `./dataset` directory.

### Extract only Python and JavaScript files

```bash
python Scripts/extract_code_samples.py --include-langs Python,JavaScript
```

### Output in all supported formats

```bash
python Scripts/extract_code_samples.py --export-format all
```

### Focus on specific directories

```bash
python Scripts/extract_code_samples.py --include-dirs src,lib,app
```

## Output

The script generates the following:

1. `code_samples_dataset.{json|csv|jsonl}`: The main dataset file in the specified format(s)
2. `dataset_metadata.json`: Metadata about the extraction process
3. `by_language/`: Directory containing separate files for each language

## Dataset Structure

Each code sample includes:

- `language`: Detected programming language
- `file_path`: Relative path to the file
- `content`: The code content
- `line_count`: Number of lines of code
- `code_type`: Type of code (class, function, unknown)
- `code_name`: Name of the class or function (if detected)
- `file_size_bytes`: Size of the file in bytes
- `extraction_timestamp`: When the sample was extracted
