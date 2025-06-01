# CI Pipeline Dependency Fixes

## Problem Description

The CI pipeline was failing during the "Install dependencies" step with PyYAML build errors:
- `subprocess-exited-with-error` during PyYAML package build
- `AttributeError: cython_sources` indicating Cython dependency issues
- Deprecated License classifiers warnings

## Root Cause Analysis

The issues were caused by:

1. **Outdated build tools**: Old versions of `setuptools`, `wheel`, and `pip` that don't properly handle modern Python packaging
2. **Missing Cython**: PyYAML requires Cython for compilation but it wasn't explicitly installed
3. **Implicit PyYAML dependency**: PyYAML was being installed as a transitive dependency without version pinning

## Solutions Implemented

### 1. Enhanced Dependency Installation Sequence

Updated all CI job dependency installation steps to:

```yaml
- name: Install dependencies
  run: |
    python -m pip install --upgrade pip setuptools wheel
    pip install cython
    pip install -r requirements.txt
    pip install -r requirements-dev.txt
    # ... other dependencies
```

**Key changes:**
- Upgrade `setuptools` and `wheel` alongside `pip`
- Install `cython` before other dependencies
- Ensure consistent order across all CI jobs

### 2. Explicit PyYAML Version Pinning

Added to `requirements.txt`:
```
PyYAML>=6.0.2  # Ensure Python 3.11 compatibility
```

**Benefits:**
- Guarantees Python 3.11 compatible version
- Prevents unexpected version conflicts
- Ensures reproducible builds

### 3. Development Dependencies Enhancement

Added to `requirements-dev.txt`:
```
# Build dependencies
cython>=3.0.0  # Required for building some dependencies like PyYAML
```

### 4. Local Validation Script

Created `scripts/validate_dependencies.py` to:
- Test dependency installation sequence locally
- Validate critical package imports
- Specifically test PyYAML functionality
- Provide early detection of dependency issues

## Usage

### For Developers

Before pushing changes:
```bash
# Validate dependencies locally
make validate-deps

# Or run directly
python scripts/validate_dependencies.py
```

### For CI Pipeline

The fixes are automatically applied in all CI jobs:
- `code-quality`
- `unit-tests` 
- `integration-tests`
- `performance-tests`
- `build`

## Testing the Fixes

### Local Testing
```bash
# Clean environment test
python -m venv test_env
source test_env/bin/activate
make install
make validate-deps
```

### CI Testing
The fixes will be validated on the next push/PR to any monitored branch.

## Benefits of These Changes

1. **Reliability**: Consistent dependency installation across all environments
2. **Performance**: Faster builds due to proper tool versions
3. **Maintainability**: Clear dependency management and validation
4. **Early Detection**: Local validation prevents CI failures
5. **Documentation**: Clear understanding of build requirements

## Affected Files

- `.github/workflows/ci.yml` - All dependency installation steps
- `requirements.txt` - Added PyYAML version pinning
- `requirements-dev.txt` - Added Cython build dependency
- `Makefile` - Updated install command and added validation
- `scripts/validate_dependencies.py` - New validation script

## Future Considerations

1. **Dependency Monitoring**: Consider using Dependabot for automated updates
2. **Build Caching**: Evaluate caching Cython compilation artifacts
3. **Alternative Packages**: Monitor for faster alternatives to problematic dependencies
4. **Python Version Updates**: Test these fixes when upgrading Python versions

## Troubleshooting

If dependency issues recur:

1. **Check Python version compatibility** - Ensure all packages support the target Python version
2. **Validate build tools** - Ensure latest `pip`, `setuptools`, `wheel` versions
3. **Check for conflicts** - Run `pip check` to identify dependency conflicts
4. **Use local validation** - Run `make validate-deps` before pushing changes

## Related Issues

- PyYAML compilation issues with Python 3.11
- Cython source attribution errors
- Setuptools deprecation warnings
- Transitive dependency version conflicts 