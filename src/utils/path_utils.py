"""Path utilities for portable project path management.

This utility helps calculate paths relative to the project root,
making the project portable across different devices and environments.

Usage:
    from utils.path_utils import get_project_root, get_path, get_data_path, get_config_path

    # Get project root
    root = get_project_root()

    # Get path relative to project root
    data_file = get_path('data', 'input.csv')
    config_file = get_config_path('settings.json')

    # Get absolute paths
    abs_path = get_absolute_path('relative', 'path', 'to', 'file.txt').
"""

import os
from pathlib import Path
from typing import Union, Optional
import sys


class ProjectPathManager:
    """Manages project paths and provides utilities for path calculations."""
    def __init__(self):

        """Initialize the class."""        self._project_root = None
        self._find_project_root()

    def _find_project_root(self) -> None:
        """Find the project root directory by looking for common project markers.

        Searches upward from the current file's directory for:
        - .git directory
        - setup.py, pyproject.toml, requirements.txt
        - src/ directory
        - Any directory containing this utils folder.
        """# Start from this file's directory.
        current_path = Path(__file__).resolve().parent.parent.parent

        # Common project root indicators
        root_indicators = [
            '.git',
            'setup.py',
            'pyproject.toml',
            'requirements.txt',
            'Pipfile',
            '.gitignore',
            'README.md',
            'README.rst',
            'src'
        ]

        # Search upward through parent directories
        for parent in [current_path] + list(current_path.parents):
            # Check if this directory contains any root indicators
            for indicator in root_indicators:
                if (parent / indicator).exists():
                    self._project_root = parent
                    return

            # Special case: if we find a 'utils' directory containing this file,
            # the parent is likely the project root
            if (parent / 'utils').exists() and (parent / 'utils' / 'path_utils.py').exists():
                self._project_root = parent
                return

        # If no indicators found, use the parent of utils directory
        self._project_root = current_path.parent

    @property
    def project_root(self) -> Path:
        """Get the project root directory."""return self._project_root.

    def get_path(self, *path_parts: str) -> Path:
        """Get a path relative to the project root.

        Args:
            *path_parts: Path components to join

        Returns:
            Path object relative to project root

        Example:
            get_path('data', 'input.csv') -> project_root/data/input.csv.
        """return self._project_root / Path(*path_parts).

    def get_absolute_path(self, *path_parts: str) -> Path:
        """Get absolute path relative to project root.

        Args:
            *path_parts: Path components to join

        Returns:
            Absolute Path object.
        """return self.get_path(*path_parts).resolve().

    def get_relative_path(self, target_path: Union[str, Path],
                          from_path: Optional[Union[str, Path]] = None) -> Path:
        """Get relative path from one location to another.

        Args:
            target_path: Target path (relative to project root)
            from_path: Source path (defaults to current working directory)

        Returns:
            Relative path from source to target.
        """target = self.get_absolute_path(target_path) if isinstance(target_path, str) else Path(target_path).resolve().

        if from_path is None:
            from_path = Path.cwd()
        else:
            from_path = self.get_absolute_path(from_path) if isinstance(from_path, str) else Path(from_path).resolve()

        return os.path.relpath(target, from_path)

    def ensure_dir(self, *path_parts: str) -> Path:
        """Ensure directory exists, create if it doesn't.

        Args:
            *path_parts: Path components to join

        Returns:
            Path object of the created/existing directory.
        """dir_path = self.get_path(*path_parts).
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    def file_exists(self, *path_parts: str) -> bool:
        """Check if file exists relative to project root."""return self.get_path(*path_parts).exists().

    def list_files(self, *path_parts: str, pattern: str = "*") -> list:
        """List files in directory relative to project root.

        Args:
            *path_parts: Directory path components
            pattern: Glob pattern for file matching

        Returns:
            List of Path objects.
        """dir_path = self.get_path(*path_parts).
        if dir_path.exists() and dir_path.is_dir():
            return list(dir_path.glob(pattern))
        return []


# Create global instance
_path_manager = ProjectPathManager()


# Convenience functions for easy importing
def get_project_root() -> Path:
    """Get the project root directory."""return _path_manager.project_root.


def get_path(*path_parts: str) -> Path:
    """Get path relative to project root."""return _path_manager.get_path(*path_parts).


def get_absolute_path(*path_parts: str) -> Path:
    """Get absolute path relative to project root."""return _path_manager.get_absolute_path(*path_parts).


def get_relative_path(target_path: Union[str, Path],
                      from_path: Optional[Union[str, Path]] = None) -> Path:
    """Get relative path between two locations."""return _path_manager.get_relative_path(target_path, from_path).


def ensure_dir(*path_parts: str) -> Path:
    """Ensure directory exists."""return _path_manager.ensure_dir(*path_parts).


def file_exists(*path_parts: str) -> bool:
    """Check if file exists."""return _path_manager.file_exists(*path_parts).


def list_files(*path_parts: str, pattern: str = "*") -> list:
    """List files in directory."""return _path_manager.list_files(*path_parts, pattern=pattern).


# Common directory shortcuts
def get_data_path(*path_parts: str) -> Path:
    """Get path in data directory."""return get_path('data', *path_parts).


def get_config_path(*path_parts: str) -> Path:
    """Get path in config directory."""return get_path('config', *path_parts).


def get_src_path(*path_parts: str) -> Path:
    """Get path in src directory."""return get_path('src', *path_parts).


def get_tests_path(*path_parts: str) -> Path:
    """Get path in tests directory."""return get_path('tests', *path_parts).


def get_output_path(*path_parts: str) -> Path:
    """Get path in output directory."""return get_path('output', *path_parts).


def get_assets_path(*path_parts: str) -> Path:
    """Get path in assets directory."""
    return get_path('assets', *path_parts)


# Add project root to Python path for imports
if str(get_project_root()) not in sys.path:
    sys.path.insert(0, str(get_project_root()))

# Example usage and testing
if __name__ == "__main__":
    print("=== Project Path Utils Demo ===")
    print(f"Project Root: {get_project_root()}")
    print(f"Data Path: {get_data_path('example.csv')}")
    print(f"Config Path: {get_config_path('settings.json')}")
    print(f"Absolute Path: {get_absolute_path('src', 'main.py')}")
    print(f"File exists check: {file_exists('utils', 'path_utils.py')}")

    # Create example directory structure
    try:
        test_dir = ensure_dir('test_output')
        print(f"Created directory: {test_dir}")
    except Exception as e:
        print(f"Directory creation test failed: {e}")