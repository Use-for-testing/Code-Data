#!/usr/bin/env python3
"""
GitHub Code Dataset Analyzer

This script loads and analyzes datasets created by the extract_code_samples.py script,
generating visualizations and statistics about the code.

Usage:
    python analyze_code_dataset.py [--dataset DATASET_PATH] [--output OUTPUT_DIR]

Requirements:
    - matplotlib
    - pandas
    - seaborn (optional, for enhanced visualizations)
"""

import os
import json
import argparse
import datetime
from pathlib import Path
import csv
import re
from collections import Counter, defaultdict

# Try to import visualization libraries, but make them optional
try:
    import matplotlib.pyplot as plt
    import pandas as pd
    VISUALIZATION_AVAILABLE = True
    
    try:
        import seaborn as sns
        sns.set_style("whitegrid")
        SEABORN_AVAILABLE = True
    except ImportError:
        SEABORN_AVAILABLE = False
        print("Seaborn not found. Basic visualizations will be used.")
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Matplotlib and/or pandas not found. Visualizations will be disabled.")
    print("To enable visualizations, install required packages:")
    print("pip install matplotlib pandas seaborn")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze code samples dataset")
    parser.add_argument("--dataset", type=str, default="./dataset",
                        help="Path to the dataset directory (default: ./dataset)")
    parser.add_argument("--output", type=str, default="./dataset/analysis",
                        help="Output directory for analysis results (default: ./dataset/analysis)")
    parser.add_argument("--format", type=str, choices=["json", "csv", "jsonl"], default="json",
                        help="Dataset format to analyze (default: json)")
    parser.add_argument("--no-vis", action="store_true",
                        help="Disable visualizations even if dependencies are available")
    return parser.parse_args()

def load_dataset(args):
    """Load the dataset from the specified path and format."""
    dataset_path = Path(args.dataset)
    
    # Determine the dataset file path based on format
    if args.format == "json":
        dataset_file = dataset_path / "code_samples_dataset.json"
    elif args.format == "csv":
        dataset_file = dataset_path / "code_samples_dataset.csv"
    elif args.format == "jsonl":
        dataset_file = dataset_path / "code_samples_dataset.jsonl"
    
    # Check if the dataset file exists
    if not dataset_file.exists():
        print(f"Error: Dataset file not found at {dataset_file}")
        return None, None
    
    # Load the dataset
    samples = []
    metadata = None
    
    try:
        if args.format == "json":
            with open(dataset_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                samples = data.get("samples", [])
                metadata = data.get("metadata", {})
                
            # Try to load metadata from separate file if not found in the main file
            if not metadata:
                metadata_file = dataset_path / "dataset_metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        
        elif args.format == "csv":
            with open(dataset_file, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                samples = list(reader)
                
            # Try to load metadata from separate file
            metadata_file = dataset_path / "dataset_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    
        elif args.format == "jsonl":
            with open(dataset_file, 'r', encoding='utf-8') as f:
                samples = [json.loads(line) for line in f if line.strip()]
                
            # Try to load metadata from separate file
            metadata_file = dataset_path / "dataset_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
    
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None
        
    return samples, metadata

def convert_to_dataframe(samples):
    """Convert samples to a pandas DataFrame if pandas is available."""
    if not VISUALIZATION_AVAILABLE:
        return None
        
    # For CSV datasets, samples might already have the correct structure
    # For JSON/JSONL, we need to extract the content and flatten the structure
    processed_samples = []
    
    for sample in samples:
        # Check if 'content' is in the sample, if not it's probably already processed
        if 'content' in sample:
            # Create a copy without the content field (to avoid huge DataFrame)
            processed_sample = {k: v for k, v in sample.items() if k != 'content'}
            
            # Calculate additional metrics
            content = sample['content']
            lines = content.split('\n')
            
            # Calculate complexity metrics
            processed_sample['char_count'] = len(content)
            processed_sample['line_count'] = len(lines)
            processed_sample['avg_line_length'] = len(content) / max(len(lines), 1)
            
            # Calculate comment metrics for common languages
            comment_count = 0
            if sample['language'] in ['Python', 'Ruby', 'Shell', 'Bash']:
                comment_count = sum(1 for line in lines if line.strip().startswith('#'))
            elif sample['language'] in ['JavaScript', 'TypeScript', 'Java', 'C', 'C++', 'C#']:
                comment_count = sum(1 for line in lines if line.strip().startswith('//'))
                # Also count /* */ comments
                comment_count += content.count('/*')
            
            processed_sample['comment_count'] = comment_count
            processed_sample['comment_ratio'] = comment_count / max(len(lines), 1)
            
            processed_samples.append(processed_sample)
        else:
            # Convert numeric fields if they're stored as strings (common in CSV)
            for field in ['line_count', 'file_size_bytes']:
                if field in sample and isinstance(sample[field], str):
                    try:
                        sample[field] = int(sample[field])
                    except (ValueError, TypeError):
                        pass
                        
            processed_samples.append(sample)
    
    # Convert to DataFrame
    df = pd.DataFrame(processed_samples)
    
    # Convert timestamps to datetime if present
    if 'extraction_timestamp' in df.columns:
        try:
            df['extraction_timestamp'] = pd.to_datetime(df['extraction_timestamp'])
        except Exception:
            pass
            
    return df

def generate_basic_statistics(samples, metadata=None):
    """Generate basic statistics from the dataset."""
    if not samples:
        return {}
        
    # Count samples by language
    languages = Counter(sample.get('language', 'Unknown') for sample in samples)
    
    # Calculate average file size and line count
    avg_file_size = sum(int(sample.get('file_size_bytes', 0)) for sample in samples) / len(samples)
    avg_line_count = sum(int(sample.get('line_count', 0)) for sample in samples) / len(samples)
    
    # Count code types
    code_types = Counter(sample.get('code_type', 'unknown') for sample in samples)
    
    # Calculate statistics by language
    lang_stats = defaultdict(lambda: {
        'count': 0, 
        'avg_size': 0, 
        'avg_lines': 0, 
        'file_paths': []
    })
    
    for sample in samples:
        lang = sample.get('language', 'Unknown')
        lang_stats[lang]['count'] += 1
        lang_stats[lang]['avg_size'] += int(sample.get('file_size_bytes', 0))
        lang_stats[lang]['avg_lines'] += int(sample.get('line_count', 0))
        lang_stats[lang]['file_paths'].append(sample.get('file_path', ''))
    
    # Calculate averages
    for lang, stats in lang_stats.items():
        stats['avg_size'] /= stats['count']
        stats['avg_lines'] /= stats['count']
        # Limit the number of file paths to avoid massive output
        stats['file_paths'] = stats['file_paths'][:5]
    
    # Top languages by sample count
    top_languages = languages.most_common(10)
    
    # Largest files
    largest_files = sorted(
        [
            (sample.get('file_path', ''), 
             int(sample.get('file_size_bytes', 0)),
             sample.get('language', 'Unknown'))
        for sample in samples
        ],
        key=lambda x: x[1],
        reverse=True
    )[:10]
    
    # Most complex files (by line count)
    most_complex_files = sorted(
        [
            (sample.get('file_path', ''), 
             int(sample.get('line_count', 0)),
             sample.get('language', 'Unknown'))
        for sample in samples
        ],
        key=lambda x: x[1],
        reverse=True
    )[:10]
    
    # Combine all statistics
    stats = {
        'total_samples': len(samples),
        'languages': dict(languages),
        'top_languages': dict(top_languages),
        'avg_file_size_bytes': avg_file_size,
        'avg_line_count': avg_line_count,
        'code_types': dict(code_types),
        'language_stats': {k: v for k, v in lang_stats.items()},
        'largest_files': largest_files,
        'most_complex_files': most_complex_files,
        'analysis_timestamp': datetime.datetime.now().isoformat()
    }
    
    # Add metadata if available
    if metadata:
        stats['original_metadata'] = metadata
    
    return stats

def create_visualizations(df, output_dir):
    """Create visualizations from the data."""
    if not VISUALIZATION_AVAILABLE or df is None or df.empty:
        return False
        
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Language Distribution (Bar Chart)
    plt.figure(figsize=(12, 8))
    if SEABORN_AVAILABLE:
        ax = sns.countplot(y='language', data=df, order=df['language'].value_counts().index[:15])
        ax.set_title('Top 15 Programming Languages in the Repository', fontsize=16)
    else:
        language_counts = df['language'].value_counts().head(15)
        language_counts.plot(kind='barh')
        plt.title('Top 15 Programming Languages in the Repository', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'language_distribution.png'))
    plt.close()
    
    # 2. File Size Distribution
    plt.figure(figsize=(10, 6))
    if 'file_size_bytes' in df.columns:
        if SEABORN_AVAILABLE:
            sns.histplot(data=df, x='file_size_bytes', bins=50, log_scale=True)
        else:
            plt.hist(df['file_size_bytes'], bins=50, log=True)
        plt.title('File Size Distribution (log scale)', fontsize=16)
        plt.xlabel('File Size (bytes)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'file_size_distribution.png'))
    plt.close()
    
    # 3. Line Count Distribution
    plt.figure(figsize=(10, 6))
    if 'line_count' in df.columns:
        if SEABORN_AVAILABLE:
            sns.histplot(data=df, x='line_count', bins=50)
        else:
            plt.hist(df['line_count'], bins=50)
        plt.title('Line Count Distribution', fontsize=16)
        plt.xlabel('Number of Lines')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'line_count_distribution.png'))
    plt.close()
    
    # 4. Average File Size by Language
    plt.figure(figsize=(12, 8))
    if 'file_size_bytes' in df.columns and 'language' in df.columns:
        avg_size_by_lang = df.groupby('language')['file_size_bytes'].mean().sort_values(ascending=False).head(15)
        avg_size_by_lang.plot(kind='barh')
        plt.title('Average File Size by Language (Top 15)', fontsize=16)
        plt.xlabel('Average Size (bytes)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'avg_size_by_language.png'))
    plt.close()
    
    # 5. Average Line Count by Language
    plt.figure(figsize=(12, 8))
    if 'line_count' in df.columns and 'language' in df.columns:
        avg_lines_by_lang = df.groupby('language')['line_count'].mean().sort_values(ascending=False).head(15)
        avg_lines_by_lang.plot(kind='barh')
        plt.title('Average Line Count by Language (Top 15)', fontsize=16)
        plt.xlabel('Average Line Count')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'avg_lines_by_language.png'))
    plt.close()
    
    # 6. Code Type Distribution
    plt.figure(figsize=(10, 6))
    if 'code_type' in df.columns:
        if SEABORN_AVAILABLE:
            sns.countplot(x='code_type', data=df)
        else:
            df['code_type'].value_counts().plot(kind='bar')
        plt.title('Code Type Distribution', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'code_type_distribution.png'))
    plt.close()
    
    # 7. Comment Ratio by Language (if available)
    if 'comment_ratio' in df.columns and 'language' in df.columns:
        plt.figure(figsize=(12, 8))
        avg_comment_ratio = df.groupby('language')['comment_ratio'].mean().sort_values(ascending=False).head(15)
        avg_comment_ratio.plot(kind='barh')
        plt.title('Average Comment Ratio by Language (Top 15)', fontsize=16)
        plt.xlabel('Average Comment Ratio (comments per line)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'comment_ratio_by_language.png'))
        plt.close()
    
    return True

def generate_html_report(stats, visualization_created, output_dir):
    """Generate an HTML report of the analysis results."""
    output_path = os.path.join(output_dir, 'code_analysis_report.html')
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Code Repository Analysis Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            h1, h2, h3 {{
                color: #0066cc;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 20px;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .visualization {{
                max-width: 100%;
                margin: 20px 0;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }}
            .stats-container {{
                display: flex;
                flex-wrap: wrap;
                justify-content: space-between;
            }}
            .stat-box {{
                background-color: #f8f9fa;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                width: 23%;
            }}
            .stat-box h3 {{
                margin-top: 0;
                border-bottom: 1px solid #ddd;
                padding-bottom: 5px;
            }}
            @media (max-width: 768px) {{
                .stat-box {{
                    width: 48%;
                }}
            }}
            @media (max-width: 480px) {{
                .stat-box {{
                    width: 100%;
                }}
            }}
        </style>
    </head>
    <body>
        <h1>Code Repository Analysis Report</h1>
        <p>Analysis performed on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Overview</h2>
        <div class="stats-container">
            <div class="stat-box">
                <h3>Total Samples</h3>
                <p>{stats.get('total_samples', 0)}</p>
            </div>
            <div class="stat-box">
                <h3>Languages</h3>
                <p>{len(stats.get('languages', {}))}</p>
            </div>
            <div class="stat-box">
                <h3>Avg File Size</h3>
                <p>{stats.get('avg_file_size_bytes', 0):.2f} bytes</p>
            </div>
            <div class="stat-box">
                <h3>Avg Line Count</h3>
                <p>{stats.get('avg_line_count', 0):.2f}</p>
            </div>
        </div>
    """
    
    # Add visualizations if created
    if visualization_created:
        html += """
        <h2>Visualizations</h2>
        <div class="visualizations-container">
            <h3>Language Distribution</h3>
            <img class="visualization" src="language_distribution.png" alt="Language Distribution">
            
            <h3>File Size Distribution</h3>
            <img class="visualization" src="file_size_distribution.png" alt="File Size Distribution">
            
            <h3>Line Count Distribution</h3>
            <img class="visualization" src="line_count_distribution.png" alt="Line Count Distribution">
            
            <h3>Average File Size by Language</h3>
            <img class="visualization" src="avg_size_by_language.png" alt="Average File Size by Language">
            
            <h3>Average Line Count by Language</h3>
            <img class="visualization" src="avg_lines_by_language.png" alt="Average Line Count by Language">
            
            <h3>Code Type Distribution</h3>
            <img class="visualization" src="code_type_distribution.png" alt="Code Type Distribution">
        """
        
        # Add comment ratio visualization if available
        comment_ratio_img = os.path.join(output_dir, 'comment_ratio_by_language.png')
        if os.path.exists(comment_ratio_img):
            html += """
            <h3>Comment Ratio by Language</h3>
            <img class="visualization" src="comment_ratio_by_language.png" alt="Comment Ratio by Language">
            """
            
        html += "</div>"
    
    # Top languages
    html += """
        <h2>Top Languages</h2>
        <table>
            <tr>
                <th>Language</th>
                <th>Sample Count</th>
                <th>Percentage</th>
            </tr>
    """
    
    total_samples = stats.get('total_samples', 0)
    for lang, count in stats.get('top_languages', {}).items():
        percentage = (count / total_samples) * 100 if total_samples > 0 else 0
        html += f"""
            <tr>
                <td>{lang}</td>
                <td>{count}</td>
                <td>{percentage:.2f}%</td>
            </tr>
        """
    
    html += "</table>"
    
    # Largest files
    html += """
        <h2>Largest Files</h2>
        <table>
            <tr>
                <th>File Path</th>
                <th>Size (bytes)</th>
                <th>Language</th>
            </tr>
    """
    
    for file_path, size, lang in stats.get('largest_files', []):
        html += f"""
            <tr>
                <td>{file_path}</td>
                <td>{size}</td>
                <td>{lang}</td>
            </tr>
        """
    
    html += "</table>"
    
    # Most complex files
    html += """
        <h2>Most Complex Files (by Line Count)</h2>
        <table>
            <tr>
                <th>File Path</th>
                <th>Line Count</th>
                <th>Language</th>
            </tr>
    """
    
    for file_path, line_count, lang in stats.get('most_complex_files', []):
        html += f"""
            <tr>
                <td>{file_path}</td>
                <td>{line_count}</td>
                <td>{lang}</td>
            </tr>
        """
    
    html += "</table>"
    
    # Language statistics
    html += """
        <h2>Language Statistics</h2>
        <table>
            <tr>
                <th>Language</th>
                <th>Count</th>
                <th>Avg Size (bytes)</th>
                <th>Avg Lines</th>
                <th>Sample File Paths</th>
            </tr>
    """
    
    # Sort languages by count
    sorted_langs = sorted(
        stats.get('language_stats', {}).items(),
        key=lambda x: x[1]['count'],
        reverse=True
    )
    
    for lang, lang_stats in sorted_langs:
        file_paths = ", ".join(lang_stats.get('file_paths', []))
        html += f"""
            <tr>
                <td>{lang}</td>
                <td>{lang_stats.get('count', 0)}</td>
                <td>{lang_stats.get('avg_size', 0):.2f}</td>
                <td>{lang_stats.get('avg_lines', 0):.2f}</td>
                <td>{file_paths}</td>
            </tr>
        """
    
    html += "</table>"
    
    # Code type distribution
    html += """
        <h2>Code Type Distribution</h2>
        <table>
            <tr>
                <th>Code Type</th>
                <th>Count</th>
                <th>Percentage</th>
            </tr>
    """
    
    code_types = stats.get('code_types', {})
    for code_type, count in code_types.items():
        percentage = (count / total_samples) * 100 if total_samples > 0 else 0
        html += f"""
            <tr>
                <td>{code_type}</td>
                <td>{count}</td>
                <td>{percentage:.2f}%</td>
            </tr>
        """
    
    html += """
        </table>
        <footer>
            <p>Generated by GitHub Code Dataset Analyzer</p>
        </footer>
    </body>
    </html>
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    return output_path

def export_statistics(stats, output_dir):
    """Export statistics to file."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Export as JSON
    json_path = os.path.join(output_dir, 'code_analysis_stats.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    # Export as CSV (flatten the top-level stats)
    csv_path = os.path.join(output_dir, 'language_stats.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['language', 'count', 'avg_size', 'avg_lines']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for lang, lang_stats in stats.get('language_stats', {}).items():
            writer.writerow({
                'language': lang,
                'count': lang_stats.get('count', 0),
                'avg_size': lang_stats.get('avg_size', 0),
                'avg_lines': lang_stats.get('avg_lines', 0)
            })
    
    return json_path, csv_path

def main():
    """Main function to run the code analysis."""
    args = parse_arguments()
    
    print("Starting GitHub Code Dataset Analyzer")
    print("------------------------------------")
    
    # Load the dataset
    print(f"\nLoading dataset from: {args.dataset}")
    samples, metadata = load_dataset(args)
    
    if not samples:
        print("No samples found or error loading dataset.")
        return
    
    print(f"Loaded {len(samples)} code samples.")
    
    # Generate statistics
    print("\nGenerating statistics...")
    stats = generate_basic_statistics(samples, metadata)
    
    if not stats:
        print("Error generating statistics.")
        return
    
    # Output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to DataFrame for analysis (if pandas is available)
    df = convert_to_dataframe(samples) if VISUALIZATION_AVAILABLE and not args.no_vis else None
    
    # Create visualizations if possible
    vis_created = False
    if VISUALIZATION_AVAILABLE and df is not None and not args.no_vis:
        print("\nGenerating visualizations...")
        vis_created = create_visualizations(df, output_dir)
    
    # Export statistics
    print("\nExporting analysis results...")
    json_path, csv_path = export_statistics(stats, output_dir)
    
    # Generate HTML report
    html_path = generate_html_report(stats, vis_created, output_dir)
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    print(f"  - Statistics: {json_path}")
    print(f"  - Language stats: {csv_path}")
    print(f"  - HTML report: {html_path}")
    
    if vis_created:
        print(f"  - Visualizations: {output_dir}/*.png")

if __name__ == "__main__":
    main()
