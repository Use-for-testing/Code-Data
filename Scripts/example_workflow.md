# Example Workflow: Using the Code Extraction Toolkit

This document demonstrates typical usage patterns for the GitHub code extraction toolkit. Follow these examples to make the most of the tools in various scenarios.

## Basic Usage: Extracting and Analyzing the Current Repository

The simplest way to use the toolkit is to run the combined script, which extracts code samples and generates an analysis:

```bash
# On Linux/Mac:
./Scripts/run_extraction_and_analysis.sh

# On Windows:
Scripts\run_extraction_and_analysis.bat
```

This will:
1. Extract code samples from the current repository
2. Save them to the `./dataset` directory
3. Generate an analysis report at `./dataset/analysis/code_analysis_report.html`

## Advanced Usage: Customizing Extraction Parameters

You can customize the extraction process with additional parameters:

```bash
./Scripts/run_extraction_and_analysis.sh \
  --repo-path ../another-repository \
  --output ./custom-dataset \
  --min-lines 10 \
  --max-samples 500 \
  --export-format all
```

This customized command:
- Extracts from a different repository
- Saves results to a custom location
- Only includes files with at least 10 lines
- Limits to 500 samples per language
- Exports in all available formats (JSON, CSV, JSONL)

## Scenario 1: Analyzing a New GitHub Project

When you want to quickly understand a new project you've cloned:

```bash
# Clone the repository
git clone https://github.com/example/new-project.git
cd new-project

# Copy the scripts (if not already present)
cp -r /path/to/Scripts .

# Run the extraction and analysis
./Scripts/run_extraction_and_analysis.sh

# Open the report in a browser
open ./dataset/analysis/code_analysis_report.html  # Mac
# or
xdg-open ./dataset/analysis/code_analysis_report.html  # Linux
# or
start ./dataset/analysis/code_analysis_report.html  # Windows
```

## Scenario 2: Creating a Language-Specific Dataset

If you only want to extract samples from specific languages:

```bash
python Scripts/extract_code_samples.py \
  --include-langs Python,JavaScript,TypeScript \
  --output ./python-js-dataset
```

## Scenario 3: Running Only the Analysis on an Existing Dataset

If you already have a dataset and just want to generate or update the analysis:

```bash
python Scripts/analyze_code_dataset.py \
  --dataset ./existing-dataset \
  --output ./existing-dataset/new-analysis
```

## Scenario 4: Extracting Code for Machine Learning Training

When preparing a dataset for machine learning:

```bash
python Scripts/extract_code_samples.py \
  --min-lines 20 \
  --max-samples 5000 \
  --output ./training-data \
  --export-format jsonl
```

## Scenario 5: Creating Comparative Analysis of Multiple Repositories

To compare multiple repositories:

```bash
# Extract from first repository
./Scripts/run_extraction_and_analysis.sh \
  --repo-path ./repo1 \
  --output ./comparison/repo1

# Extract from second repository
./Scripts/run_extraction_and_analysis.sh \
  --repo-path ./repo2 \
  --output ./comparison/repo2

# Now manually compare the analysis reports
```

## Troubleshooting

If you encounter issues with the extraction process:

1. Check that Python 3.6+ is installed and accessible
2. Ensure dependencies are installed: `pip install -r Scripts/requirements.txt`
3. For visualization problems, confirm matplotlib, pandas, and seaborn are installed
4. For permission errors, make sure scripts are executable: `chmod +x Scripts/*.sh`
