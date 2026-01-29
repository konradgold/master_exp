# Hydra Migration Guide

This document describes the migration from argparse to Hydra configuration management.

## Status

### ✅ Completed
1. **run_training_vqvae.py** - Fully converted to Hydra
   - Config: `conf/config_vqvae.yaml`
   - Uses `@hydra.main` decorator
   - All `args` → `cfg` conversions complete

2. **run_training_4m.py** - Already using Hydra
   - Config: `conf/config_4m.yaml`
   - Already has `@hydra.main` decorator

### 🔄 In Progress
3. **run_generation.py** - Needs conversion
4. **run_training_divae.py** - Needs conversion
5. **run_training_vqcontrolnet.py** - Needs conversion
6. **run_training_4m_fsdp.py** - Needs review
7. **train_wordpiece_tokenizer.py** - Needs conversion
8. **save_vq_tokens.py** - Needs conversion

## Migration Pattern

For each script that needs conversion, follow this pattern:

### 1. Update Imports
```python
# Remove
import argparse

# Add
import hydra
from omegaconf import DictConfig, OmegaConf
```

### 2. Remove get_args() Function
Delete the entire `get_args()` function and its argument parser setup.

### 3. Create Hydra Config YAML
Create `conf/config_{script_name}.yaml` with all parameters. Example:
```yaml
# Model parameters
patch_size: 16
input_size: 224

# Training parameters  
batch_size: 256
epochs: 100
# ... etc
```

### 4. Update Function Signatures
```python
# Before
def main(args: argparse.Namespace):

# After  
@hydra.main(config_path="../conf", config_name="config_name", version_base=None)
def main(cfg: DictConfig):
    # Convert to OmegaConf for backward compatibility if needed
    cfg = OmegaConf.create(cfg)
```

### 5. Update Variable References
Replace all `args.parameter` with `cfg.parameter` throughout the file.

### 6. Update if __name__ == '__main__'
```python
# Before
if __name__ == '__main__':
    args = get_args()
    main(args)

# After
if __name__ == '__main__':
    main()
```

## Benefits of Hydra

1. **Better Organization**: Configuration separate from code
2. **Config Composition**: Can combine multiple config files
3. **Command-line Overrides**: `python script.py batch_size=128`
4. **Automatic Logging**: Hydra creates output directories with timestamps
5. **Type Safety**: Better validation and type checking
6. **Config Groups**: Easy to switch between different setups

## Usage Examples

```bash
# Use default config
python run_training_vqvae.py

# Override specific parameters
python run_training_vqvae.py batch_size=128 epochs=200 domain=depth

# Use different config file
python run_training_vqvae.py --config-name=custom_config

# Multiple overrides
python run_training_vqvae.py batch_size=64 blr=0.0002 wandb_project=my_project
```

## Next Steps

1. Convert run_generation.py to Hydra
2. Convert remaining training scripts (divae, vqcontrolnet, fsdp)
3. Convert utility scripts (train_wordpiece_tokenizer, save_vq_tokens)
4. Update documentation with Hydra usage examples
5. Test all converted scripts to ensure functionality

## Notes

- All config files should be in the `conf/` directory
- Use `version_base=None` in @hydra.main to avoid deprecation warnings
- Convert DictConfig to OmegaConf in main() for backward compatibility with existing utility functions
- Keep the same parameter names to minimize changes to other parts of the codebase
