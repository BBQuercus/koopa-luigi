# koopa-luigi
A functional implementation of koopa in luigi with multi-environment support for different model versions.

## Quick Start

### 1. Setup Environments (One Time)
```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup both environments
./setup_envs.sh
```

This creates:
- `venv_modern/` - TensorFlow 2.17+ for recent models
- `venv_legacy/` - TensorFlow 2.13 for older models (model_fish.h5)

### 2. Run the Pipeline

**For recent models:**
```bash
./run_modern.sh --config koopa.cfg --workers 4
```

**For older models (model_fish.h5):**
```bash
./run_legacy.sh --config koopa.cfg --workers 4
```

**Skip incompatible models (any environment):**
```bash
koopa-luigi --config koopa.cfg --skip-incompatible --workers 4
```

### 3. Check Environment

```bash
# Show environment info and installed packages
koopa-luigi --env-info

# Check model compatibility with your config
python check_environment.py --config koopa.cfg

# Verbose mode with detailed compatibility info
koopa-luigi --config koopa.cfg --verbose-compatibility
```

## Configuration

Use standard koopa configuration files. The pipeline automatically detects:
- Input file formats (ND2, CZI, TIFF, etc.)
- Model compatibility with current environment
- Required segmentation methods

## Running on HPC

### SLURM
```bash
# GPU workflow
sbatch gpu.sh

# CPU workflow  
sbatch cpu.sh

# Chain GPU then CPU
gpu_id=$(sbatch --parsable gpu.sh)
sbatch -d afterok:$gpu_id cpu.sh
```

### Local/Xenon (CPU only)
```bash
sh cpu.sh
```

## Troubleshooting

**Model compatibility errors:**
- Use `./run_legacy.sh` for older models
- Use `./run_modern.sh` for recent models
- Add `--skip-incompatible` to skip problematic models

**Missing image format support:**
```bash
# Install specific readers as needed
uv pip install nd2reader  # For Nikon ND2
uv pip install czifile    # For Zeiss CZI
```

## Documentation
* Slides [here](https://docs.google.com/presentation/d/1NnMhKKv6QjvK3uVa6e8yZHLHRFyKN9We8xGsiD1sIcY/edit?usp=sharing)
* Video [here](https://youtu.be/R6RBIBuJDGI)