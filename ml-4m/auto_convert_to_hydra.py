#!/usr/bin/env python3
"""
Automated script to convert argparse-based Python files to Hydra configuration.
"""

import re
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional

def extract_argparse_args(file_path: Path) -> List[Dict]:
    """Extract all argparse arguments from a Python file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    args = []
    # Find all parser.add_argument calls
    pattern = r"parser\.add_argument\([^)]+\)"
    matches = re.finditer(pattern, content, re.DOTALL)
    
    for match in matches:
        arg_text = match.group()
        arg_info = {}
        
        # Extract argument name
        name_match = re.search(r"'--([^']+)'|\"--([^\"]+)\"", arg_text)
        if name_match:
            arg_info['name'] = name_match.group(1) or name_match.group(2)
        
        # Extract default value
        default_match = re.search(r"default=([^,)]+)", arg_text)
        if default_match:
            default_val = default_match.group(1).strip()
            # Try to eval simple values
            try:
                if default_val in ['None', 'True', 'False']:
                    arg_info['default'] = eval(default_val)
                elif default_val.startswith('['):
                    arg_info['default'] = eval(default_val)
                elif default_val.startswith("'") or default_val.startswith('"'):
                    arg_info['default'] = eval(default_val)
                else:
                    try:
                        arg_info['default'] = float(default_val) if '.' in default_val else int(default_val)
                    except:
                        arg_info['default'] = default_val
            except:
                arg_info['default'] = default_val
        
        # Check if it's a boolean action
        if 'action=' in arg_text:
            if "'store_true'" in arg_text or '"store_true"' in arg_text:
                arg_info['default'] = arg_info.get('default', False)
                arg_info['type'] = 'bool'
            elif "'store_false'" in arg_text or '"store_false"' in arg_text:
                arg_info['default'] = arg_info.get('default', True)
                arg_info['type'] = 'bool'
        
        if 'name' in arg_info:
            args.append(arg_info)
    
    return args

def create_hydra_config(args: List[Dict], config_name: str) -> str:
    """Create a Hydra YAML config from extracted arguments."""
    config = {}
    for arg in args:
        if 'default' in arg:
            config[arg['name']] = arg['default']
        else:
            config[arg['name']] = None
    
    yaml_content = yaml.dump(config, default_flow_style=False, sort_keys=False)
    return yaml_content

def convert_file_to_hydra(file_path: Path, config_name: str) -> Tuple[str, str]:
    """Convert a single file from argparse to Hydra."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Step 1: Replace imports
    content = re.sub(
        r'import argparse\n',
        'import hydra\nfrom omegaconf import DictConfig, OmegaConf\n',
        content
    )
    
    # Step 2: Remove get_args function (find start and end)
    get_args_start = content.find('def get_args(')
    if get_args_start != -1:
        # Find the return statement
        return_match = re.search(r'\n    return args\n', content[get_args_start:])
        if return_match:
            get_args_end = get_args_start + return_match.end()
            # Extract args before removing
            args = extract_argparse_args(file_path)
            
            # Replace get_args with a comment
            content = (
                content[:get_args_start] +
                '# Configuration now handled via Hydra - see conf/' + config_name + '.yaml\n' +
                content[get_args_end:]
            )
    
    # Step 3: Update function signatures (args -> cfg)
    content = re.sub(
        r'def (\w+)\(args(?::[\s]*argparse\.Namespace)?\)',
        r'def \1(cfg: DictConfig)',
        content
    )
    
    # Step 4: Replace args. with cfg. 
    content = re.sub(r'\bargs\.', 'cfg.', content)
    
    # Step 5: Add @hydra.main decorator to main function
    main_func_match = re.search(r'\ndef main\(cfg: DictConfig\):', content)
    if main_func_match:
        decorator = f'@hydra.main(config_path="../conf", config_name="{config_name}", version_base=None)\n'
        content = content[:main_func_match.start()] + '\n' + decorator + content[main_func_match.start()+1:]
        
        # Convert cfg to OmegaConf for backward compatibility  
        main_body_start = main_func_match.end()
        indent = '    '
        conversion_code = f'\n{indent}# Convert DictConfig to OmegaConf for compatibility\n{indent}cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))\n'
        content = content[:main_body_start] + conversion_code + content[main_body_start:]
    
    # Step 6: Simplify if __name__ == '__main__':
    main_call_pattern = r"if __name__ == '__main__':\s+args = get_args\(\)[^}]+main\(args\)"
    content = re.sub(
        main_call_pattern,
        "if __name__ == '__main__':\n    main()",
        content,
        flags=re.DOTALL
    )
    
    # Alternative pattern
    if "if __name__ == '__main__':" in content:
        main_idx = content.find("if __name__ == '__main__':")
        if main_idx != -1:
            # Replace everything after with simple main() call
            content = content[:main_idx] + "if __name__ == '__main__':\n    main()\n"
    
    return content, args

def main():
    """Convert all remaining files to Hydra."""
    files_to_convert = [
        ('run_generation.py', 'config_generation'),
        ('run_training_divae.py', 'config_divae'),
        ('run_training_vqcontrolnet.py', 'config_vqcontrolnet'),
        ('run_training_4m_fsdp.py', 'config_4m_fsdp'),
        ('train_wordpiece_tokenizer.py', 'config_wordpiece_tokenizer'),
        ('save_vq_tokens.py', 'config_save_vq_tokens'),
    ]
    
    base_dir = Path(__file__).parent
    conf_dir = base_dir / 'conf'
    conf_dir.mkdir(exist_ok=True)
    
    for file_name, config_name in files_to_convert:
        file_path = base_dir / file_name
        if not file_path.exists():
            print(f"❌ File not found: {file_name}")
            continue
        
        print(f"\n{'='*80}")
        print(f"Converting: {file_name}")
        print(f"{'='*80}")
        
        try:
            # Extract arguments first for config file
            args = extract_argparse_args(file_path)
            print(f"  Found {len(args)} arguments")
            
            # Create YAML config
            yaml_content = create_hydra_config(args, config_name)
            yaml_path = conf_dir / f"{config_name}.yaml"
            with open(yaml_path, 'w') as f:
                f.write(yaml_content)
            print(f"  ✅ Created config: {yaml_path}")
            
            # Convert Python file
            new_content, _ = convert_file_to_hydra(file_path, config_name)
            
            # Backup original
            backup_path = file_path.with_suffix('.py.argparse_backup')
            with open(backup_path, 'w') as f:
                with open(file_path, 'r') as orig:
                    f.write(orig.read())
            print(f"  📦 Backed up original to: {backup_path}")
            
            # Write converted file
            with open(file_path, 'w') as f:
                f.write(new_content)
            print(f"  ✅ Converted: {file_path}")
            
        except Exception as e:
            print(f"  ❌ Error converting {file_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("Conversion complete!")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
