#!/usr/bin/env python3
"""
Test Script for GitHub Code Extraction Toolkit

This script performs basic tests to verify that the code extraction toolkit
is functioning correctly. It creates a temporary test environment, runs the
extraction and analysis tools, and verifies the expected outputs.

Usage:
    python test_toolkit.py
"""

import os
import sys
import tempfile
import shutil
import subprocess
import unittest
import json
from pathlib import Path

class ToolkitTests(unittest.TestCase):
    """Test cases for the GitHub Code Extraction Toolkit."""
    
    @classmethod
    def setUpClass(cls):
        """Set up a temporary test environment with sample code files."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.scripts_dir = os.path.dirname(os.path.abspath(__file__))
        cls.output_dir = os.path.join(cls.temp_dir, "output")
        
        # Create test directory structure with sample files
        cls.create_test_files()
        
    @classmethod
    def tearDownClass(cls):
        """Clean up the temporary test environment."""
        shutil.rmtree(cls.temp_dir)
        
    @classmethod
    def create_test_files(cls):
        """Create sample code files for testing the extraction tool."""
        # Create a test repo structure
        test_repo = os.path.join(cls.temp_dir, "test_repo")
        os.makedirs(test_repo)
        
        # Create subdirectories
        for dir_name in ["src", "lib", "test", "docs"]:
            os.makedirs(os.path.join(test_repo, dir_name))
            
        # Create Python file
        with open(os.path.join(test_repo, "src", "main.py"), "w") as f:
            f.write("""
import os
import sys

def main():
    \"\"\"Main function for the application.\"\"\"
    print("Hello, world!")
    
if __name__ == "__main__":
    main()
""")

        # Create JavaScript file
        with open(os.path.join(test_repo, "src", "app.js"), "w") as f:
            f.write("""
// Simple JavaScript file
function showMessage() {
    console.log("Hello from JavaScript!");
    return true;
}

// Call the function
showMessage();
""")

        # Create C++ file
        with open(os.path.join(test_repo, "lib", "utils.cpp"), "w") as f:
            f.write("""
#include <iostream>
#include <string>

class Utils {
public:
    static void printMessage(const std::string& message) {
        std::cout << message << std::endl;
    }
};

int main() {
    Utils::printMessage("Hello from C++!");
    return 0;
}
""")

        # Create HTML file
        with open(os.path.join(test_repo, "index.html"), "w") as f:
            f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Test Page</title>
</head>
<body>
    <h1>Hello World</h1>
    <p>This is a test HTML file.</p>
    <script src="src/app.js"></script>
</body>
</html>
""")

    def test_extract_code_samples(self):
        """Test the basic functionality of extract_code_samples.py."""
        # Run the extraction script
        extract_script = os.path.join(self.scripts_dir, "extract_code_samples.py")
        test_repo = os.path.join(self.temp_dir, "test_repo")
        
        cmd = [
            sys.executable,
            extract_script,
            "--repo-path", test_repo,
            "--output", self.output_dir,
            "--min-lines", "3"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, f"Extraction failed with error: {result.stderr}")
        
        # Check if output files exist
        json_output = os.path.join(self.output_dir, "code_samples_dataset.json")
        self.assertTrue(os.path.exists(json_output), "JSON output file not created")
        
        # Verify JSON content
        with open(json_output, "r") as f:
            data = json.load(f)
            self.assertIn("samples", data, "No samples in JSON output")
            self.assertIn("metadata", data, "No metadata in JSON output")
            self.assertTrue(len(data["samples"]) > 0, "No code samples extracted")
            
            # Check if all test files were extracted
            languages = {sample["language"] for sample in data["samples"]}
            self.assertTrue("Python" in languages, "Python sample not extracted")
            self.assertTrue("JavaScript" in languages, "JavaScript sample not extracted")
            self.assertTrue("C++" in languages, "C++ sample not extracted")
            self.assertTrue("HTML" in languages, "HTML sample not extracted")
    
    def test_analyze_code_dataset(self):
        """Test the basic functionality of analyze_code_dataset.py."""
        # First run extraction if needed
        if not os.path.exists(os.path.join(self.output_dir, "code_samples_dataset.json")):
            self.test_extract_code_samples()
            
        # Run the analysis script
        analysis_script = os.path.join(self.scripts_dir, "analyze_code_dataset.py")
        analysis_output = os.path.join(self.output_dir, "analysis")
        
        # Check if matplotlib is available
        try:
            import matplotlib
            has_matplotlib = True
        except ImportError:
            has_matplotlib = False
            
        cmd = [
            sys.executable,
            analysis_script,
            "--dataset", self.output_dir,
            "--output", analysis_output
        ]
        
        if not has_matplotlib:
            cmd.append("--no-vis")
            
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, f"Analysis failed with error: {result.stderr}")
        
        # Check if output files exist
        stats_output = os.path.join(analysis_output, "code_analysis_stats.json")
        html_output = os.path.join(analysis_output, "code_analysis_report.html")
        
        self.assertTrue(os.path.exists(stats_output), "Stats output file not created")
        self.assertTrue(os.path.exists(html_output), "HTML report not created")
        
        # Verify stats content
        with open(stats_output, "r") as f:
            stats = json.load(f)
            self.assertIn("total_samples", stats, "No total_samples in stats output")
            self.assertIn("languages", stats, "No languages in stats output")

def main():
    """Run the tests."""
    print("Testing GitHub Code Extraction Toolkit...")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    print("\nTests completed!")

if __name__ == "__main__":
    main()
