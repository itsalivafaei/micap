#!/usr/bin/env python3
"""
Docstring Coverage Checker for MICAP
Scans the codebase and reports on missing docstrings and documentation coverage.
"""

import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
from collections import defaultdict


class DocstringChecker:
    """
    Analyzes Python files for docstring coverage and quality.
    """

    def __init__(self, source_dir: str = "src"):
        """
        Initialize the docstring checker.
        
        Args:
            source_dir: Directory to scan for Python files
        """
        self.source_dir = Path(source_dir)
        self.stats = defaultdict(int)
        self.missing_docstrings = []
        self.files_analyzed = []

    def check_file(self, file_path: Path) -> Dict[str, List[str]]:
        """
        Check a single Python file for docstring coverage.
        
        Args:
            file_path: Path to the Python file to analyze
            
        Returns:
            Dictionary containing missing docstrings by category
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            missing = {
                'module': [],
                'classes': [],
                'functions': [],
                'methods': []
            }
            
            # Check module docstring
            if not ast.get_docstring(tree):
                missing['module'].append(str(file_path))
                self.stats['missing_module_docstrings'] += 1
            else:
                self.stats['module_docstrings'] += 1
            
            # Walk the AST
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    self.stats['total_classes'] += 1
                    if not ast.get_docstring(node):
                        missing['classes'].append(f"{file_path}:{node.lineno} - {node.name}")
                        self.stats['missing_class_docstrings'] += 1
                    else:
                        self.stats['class_docstrings'] += 1
                        
                elif isinstance(node, ast.FunctionDef):
                    # Skip private methods and special methods for now
                    if not node.name.startswith('_'):
                        self.stats['total_functions'] += 1
                        if not ast.get_docstring(node):
                            # Determine if it's a method or function
                            parent = getattr(node, 'parent', None)
                            if hasattr(node, 'parent') and isinstance(node.parent, ast.ClassDef):
                                missing['methods'].append(f"{file_path}:{node.lineno} - {node.name}")
                                self.stats['missing_method_docstrings'] += 1
                            else:
                                missing['functions'].append(f"{file_path}:{node.lineno} - {node.name}")
                                self.stats['missing_function_docstrings'] += 1
                        else:
                            self.stats['function_docstrings'] += 1
            
            return missing
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return {'module': [], 'classes': [], 'functions': [], 'methods': []}

    def scan_directory(self) -> None:
        """
        Scan the source directory for Python files and check docstrings.
        """
        print(f"Scanning {self.source_dir} for Python files...")
        
        python_files = list(self.source_dir.rglob("*.py"))
        print(f"Found {len(python_files)} Python files")
        
        for file_path in python_files:
            # Skip __pycache__ and other non-source files
            if '__pycache__' in str(file_path) or 'test_' in file_path.name:
                continue
                
            self.files_analyzed.append(file_path)
            missing = self.check_file(file_path)
            
            # Collect all missing docstrings
            for category, items in missing.items():
                self.missing_docstrings.extend([(category, item) for item in items])

    def generate_report(self) -> str:
        """
        Generate a comprehensive docstring coverage report.
        
        Returns:
            Formatted report string
        """
        total_items = (
            self.stats['total_classes'] + 
            self.stats['total_functions'] + 
            len(self.files_analyzed)
        )
        
        documented_items = (
            self.stats['class_docstrings'] + 
            self.stats['function_docstrings'] + 
            self.stats['module_docstrings']
        )
        
        coverage_percentage = (documented_items / total_items * 100) if total_items > 0 else 0
        
        report = f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                           DOCSTRING COVERAGE REPORT                          ║
╚═══════════════════════════════════════════════════════════════════════════════╝

📊 SUMMARY STATISTICS:
├─ Files analyzed: {len(self.files_analyzed)}
├─ Total modules: {len(self.files_analyzed)}
├─ Total classes: {self.stats['total_classes']}
├─ Total functions: {self.stats['total_functions']}
└─ Overall coverage: {coverage_percentage:.1f}%

📈 DOCSTRING COVERAGE:
├─ Module docstrings: {self.stats['module_docstrings']}/{len(self.files_analyzed)} ({self.stats['module_docstrings']/len(self.files_analyzed)*100:.1f}%)
├─ Class docstrings: {self.stats['class_docstrings']}/{self.stats['total_classes']} ({self.stats['class_docstrings']/max(self.stats['total_classes'], 1)*100:.1f}%)
└─ Function docstrings: {self.stats['function_docstrings']}/{self.stats['total_functions']} ({self.stats['function_docstrings']/max(self.stats['total_functions'], 1)*100:.1f}%)

❌ MISSING DOCSTRINGS:
"""
        
        # Group missing docstrings by category
        missing_by_category = defaultdict(list)
        for category, item in self.missing_docstrings:
            missing_by_category[category].append(item)
        
        for category, items in missing_by_category.items():
            if items:
                report += f"\n📁 {category.upper()}:\n"
                for item in sorted(items):
                    report += f"   • {item}\n"
        
        # Recommendations
        report += f"""
💡 RECOMMENDATIONS:
"""
        if coverage_percentage < 80:
            report += "   • Add docstrings to improve coverage above 80%\n"
        if self.stats['missing_module_docstrings'] > 0:
            report += "   • Add module-level docstrings describing file purpose\n"
        if self.stats['missing_class_docstrings'] > 0:
            report += "   • Add class docstrings describing functionality and usage\n"
        if self.stats['missing_function_docstrings'] > 0:
            report += "   • Add function docstrings with Args, Returns, and Examples\n"
        
        if coverage_percentage >= 80:
            report += "   ✅ Great job! Documentation coverage is above 80%\n"
        
        return report

    def save_report(self, output_file: str = "docstring_coverage_report.txt") -> None:
        """
        Save the docstring coverage report to a file.
        
        Args:
            output_file: Path to save the report
        """
        report = self.generate_report()
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report saved to {output_file}")


def main():
    """
    Main function to run the docstring coverage checker.
    """
    parser = argparse.ArgumentParser(description="Check docstring coverage in Python codebase")
    parser.add_argument("--source-dir", default="src", help="Source directory to scan")
    parser.add_argument("--output", default="docstring_coverage_report.txt", help="Output file for report")
    parser.add_argument("--fail-under", type=float, default=80.0, help="Fail if coverage is below this percentage")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Check if source directory exists
    if not Path(args.source_dir).exists():
        print(f"Error: Source directory '{args.source_dir}' does not exist")
        sys.exit(1)
    
    # Run docstring checker
    checker = DocstringChecker(args.source_dir)
    checker.scan_directory()
    
    # Generate and display report
    report = checker.generate_report()
    print(report)
    
    # Save report
    checker.save_report(args.output)
    
    # Calculate final coverage
    total_items = (
        checker.stats['total_classes'] + 
        checker.stats['total_functions'] + 
        len(checker.files_analyzed)
    )
    
    documented_items = (
        checker.stats['class_docstrings'] + 
        checker.stats['function_docstrings'] + 
        checker.stats['module_docstrings']
    )
    
    coverage_percentage = (documented_items / total_items * 100) if total_items > 0 else 0
    
    # Exit with error code if coverage is below threshold
    if coverage_percentage < args.fail_under:
        print(f"\n❌ FAIL: Documentation coverage ({coverage_percentage:.1f}%) is below threshold ({args.fail_under}%)")
        sys.exit(1)
    else:
        print(f"\n✅ PASS: Documentation coverage ({coverage_percentage:.1f}%) meets threshold ({args.fail_under}%)")
        sys.exit(0)


if __name__ == "__main__":
    main() 