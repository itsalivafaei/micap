#!/usr/bin/env python3
"""
CI/CD Setup Validation Script for MICAP
Validates the CI/CD configuration and provides setup guidance.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Any
import json


class CICDValidator:
    """
    Validates CI/CD configuration and setup for the MICAP project.
    """

    def __init__(self):
        """Initialize the CI/CD validator."""
        self.project_root = Path.cwd()
        self.issues = []
        self.successes = []

    def check_file_exists(self, file_path: str, description: str) -> bool:
        """
        Check if a required file exists.
        
        Args:
            file_path: Path to the file to check
            description: Description of the file's purpose
            
        Returns:
            True if file exists, False otherwise
        """
        full_path = self.project_root / file_path
        if full_path.exists():
            self.successes.append(f"✅ {description}: {file_path}")
            return True
        else:
            self.issues.append(f"❌ Missing {description}: {file_path}")
            return False

    def check_github_workflows(self) -> None:
        """Check if GitHub Actions workflows are properly configured."""
        print("🔍 Checking GitHub Actions workflows...")
        
        workflows = [
            (".github/workflows/ci.yml", "Main CI/CD pipeline"),
            (".github/workflows/documentation.yml", "Documentation workflow"),
            (".github/workflows/advanced-ci.yml", "Advanced CI workflow"),
        ]
        
        for workflow_file, description in workflows:
            self.check_file_exists(workflow_file, description)

    def check_docker_config(self) -> None:
        """Check Docker configuration files."""
        print("🐳 Checking Docker configuration...")
        
        docker_files = [
            ("Dockerfile", "Docker image configuration"),
            ("docker-compose.yml", "Docker Compose configuration"),
            (".dockerignore", "Docker ignore file"),
        ]
        
        for docker_file, description in docker_files:
            self.check_file_exists(docker_file, description)

    def check_code_quality_config(self) -> None:
        """Check code quality configuration files."""
        print("🔧 Checking code quality configuration...")
        
        quality_files = [
            (".pre-commit-config.yaml", "Pre-commit hooks configuration"),
            ("pyproject.toml", "Python project configuration"),
            (".gitignore", "Git ignore configuration"),
        ]
        
        for quality_file, description in quality_files:
            self.check_file_exists(quality_file, description)

    def check_requirements_files(self) -> None:
        """Check Python requirements files."""
        print("📦 Checking Python requirements...")
        
        req_files = [
            ("requirements.txt", "Main Python dependencies"),
            ("requirements-dev.txt", "Development dependencies"),
        ]
        
        for req_file, description in req_files:
            self.check_file_exists(req_file, description)

    def check_project_structure(self) -> None:
        """Check essential project directories."""
        print("📁 Checking project structure...")
        
        directories = [
            ("src/", "Source code directory"),
            ("tests/", "Test directory"),
            ("scripts/", "Scripts directory"),
            ("config/", "Configuration directory"),
            ("data/", "Data directory"),
        ]
        
        for directory, description in directories:
            if (self.project_root / directory).exists():
                self.successes.append(f"✅ {description}: {directory}")
            else:
                self.issues.append(f"❌ Missing {description}: {directory}")

    def check_documentation(self) -> None:
        """Check documentation files."""
        print("📚 Checking documentation...")
        
        doc_files = [
            ("README.md", "Project README"),
            ("scripts/check_docstrings.py", "Docstring checker script"),
        ]
        
        for doc_file, description in doc_files:
            self.check_file_exists(doc_file, description)
        
        # Check README content
        readme_path = self.project_root / "README.md"
        if readme_path.exists():
            with open(readme_path, 'r') as f:
                content = f.read()
                if len(content) > 1000:  # Reasonable README length
                    self.successes.append("✅ README.md has substantial content")
                else:
                    self.issues.append("⚠️ README.md might need more content")

    def check_makefile(self) -> None:
        """Check if Makefile exists and is properly configured."""
        print("🔨 Checking Makefile...")
        
        if self.check_file_exists("Makefile", "Development automation"):
            makefile_path = self.project_root / "Makefile"
            with open(makefile_path, 'r') as f:
                content = f.read()
                required_targets = ['help', 'install', 'test', 'lint', 'format']
                missing_targets = []
                
                for target in required_targets:
                    if f"{target}:" not in content:
                        missing_targets.append(target)
                
                if missing_targets:
                    self.issues.append(f"⚠️ Makefile missing targets: {', '.join(missing_targets)}")
                else:
                    self.successes.append("✅ Makefile has all essential targets")

    def check_git_setup(self) -> None:
        """Check Git repository setup."""
        print("🌿 Checking Git setup...")
        
        if (self.project_root / ".git").exists():
            self.successes.append("✅ Git repository initialized")
            
            # Check if there are any commits
            try:
                result = subprocess.run(
                    ["git", "log", "--oneline", "-n", "1"],
                    capture_output=True, text=True, cwd=self.project_root
                )
                if result.returncode == 0:
                    self.successes.append("✅ Git repository has commits")
                else:
                    self.issues.append("⚠️ Git repository has no commits yet")
            except FileNotFoundError:
                self.issues.append("❌ Git command not found")
        else:
            self.issues.append("❌ Not a Git repository")

    def check_python_environment(self) -> None:
        """Check Python environment setup."""
        print("🐍 Checking Python environment...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version.major == 3 and python_version.minor >= 11:
            self.successes.append(f"✅ Python version: {python_version.major}.{python_version.minor}")
        else:
            self.issues.append(f"⚠️ Python 3.11+ recommended, found: {python_version.major}.{python_version.minor}")
        
        # Check if virtual environment is active
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            self.successes.append("✅ Virtual environment is active")
        else:
            self.issues.append("⚠️ Virtual environment not detected")

    def generate_report(self) -> str:
        """
        Generate a comprehensive validation report.
        
        Returns:
            Formatted validation report
        """
        total_checks = len(self.successes) + len(self.issues)
        success_rate = (len(self.successes) / total_checks * 100) if total_checks > 0 else 0
        
        report = f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                        MICAP CI/CD VALIDATION REPORT                         ║
╚═══════════════════════════════════════════════════════════════════════════════╝

📊 SUMMARY:
├─ Total checks: {total_checks}
├─ Successful: {len(self.successes)}
├─ Issues found: {len(self.issues)}
└─ Success rate: {success_rate:.1f}%

✅ SUCCESSFUL CHECKS:
"""
        
        for success in self.successes:
            report += f"   {success}\n"
        
        if self.issues:
            report += f"\n❌ ISSUES FOUND:\n"
            for issue in self.issues:
                report += f"   {issue}\n"
        
        report += f"""
💡 NEXT STEPS:
"""
        
        if success_rate >= 90:
            report += "   🎉 Excellent! Your CI/CD setup is nearly complete.\n"
        elif success_rate >= 75:
            report += "   👍 Good progress! Address the remaining issues.\n"
        else:
            report += "   🔧 More setup needed. Follow the installation guide.\n"
        
        if self.issues:
            report += "   📝 Review and fix the issues listed above.\n"
            report += "   📖 Consult the CI/CD documentation for guidance.\n"
        
        report += "   🚀 Run 'make install' to set up the development environment.\n"
        report += "   🧪 Run 'make test' to verify everything works.\n"
        
        return report

    def run_validation(self) -> None:
        """Run complete CI/CD validation."""
        print("🔍 Starting MICAP CI/CD validation...\n")
        
        self.check_python_environment()
        self.check_git_setup()
        self.check_project_structure()
        self.check_requirements_files()
        self.check_github_workflows()
        self.check_docker_config()
        self.check_code_quality_config()
        self.check_documentation()
        self.check_makefile()
        
        # Generate and display report
        report = self.generate_report()
        print(report)
        
        # Save report
        with open("cicd_validation_report.txt", 'w') as f:
            f.write(report)
        print("📄 Report saved to cicd_validation_report.txt")


def main():
    """Main function to run CI/CD validation."""
    print("MICAP CI/CD Setup Validator")
    print("=" * 50)
    
    validator = CICDValidator()
    validator.run_validation()
    
    # Exit with error code if there are critical issues
    critical_issues = [issue for issue in validator.issues if "❌" in issue]
    if critical_issues:
        print(f"\n⚠️ Found {len(critical_issues)} critical issues.")
        print("Please address these before proceeding with CI/CD setup.")
        sys.exit(1)
    else:
        print("\n✅ CI/CD validation completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main() 