#!/usr/bin/env python3
"""
Helper script to convert argparse-based scripts to Hydra.
This script can extract argparse arguments and generate a Hydra config file.
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def extract_argparse_args(file_path: str) -> List[Dict]:
    """Extract all argparse arguments from a Python file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find all parser.add_argument calls
    pattern = r"parser\.add_argument\((.*?)\)"
    matches = re.findall(pattern, content, re.DOTALL)
    
    args_list = []
    for match in matches:
        # Extract arg name
        name_match = re.search(r"['\"]--(\w+)['\"]", match)
        if name_match:
            arg_name = name_match.group(1)
            
            # Extract default value
            default_match = re.search(r"default=([^,\)]+)", match)
            default = default_match.group(1).strip() if default_match else None
            
            # Extract type
            type_match = re.search(r"type=(\w+)", match)
            arg_type = type_match.group(1) if type_match else "str"
            
            # Extract help
            help_match = re.search(r"help=['\"]([^'\"]+)['\"]", match)
            help_text = help_match.group(1) if help_match else ""
            
            # Check for action
            action_match = re.search(r"action=['\"]store_true['\"]", match)
            is_bool = action_match is not None
            
            args_list.append({
                'name': arg_name,
                'default': default,
                'type': arg_type,
                'help': help_text,
                'is_bool': is_bool
            })
    
    return args_list


def generate_hydra_config(args_list: List[Dict], output_path: str):
    """Generate a Hydra YAML config from extracted arguments."""
    lines = []
    lines.append("# Auto-generated Hydra configuration")
    lines.append("# Generated from argparse definitions\n")
    
    current_section = None
    for arg in args_list:
        # Try to group related parameters
        name = arg['name']
        
        # Add section comments
        if 'model' in name.lower() and current_section != 'model':
            lines.append("\n# Model parameters")
            current_section = 'model'
        elif any(x in name.lower() for x in ['batch', 'epoch', 'opt', 'lr', 'weight']) and current_section != 'training':
            lines.append("\n# Training parameters")
            current_section = 'training'
        elif 'data' in name.lower() and current_section != 'data':
            lines.append("\n# Data parameters")
            current_section = 'data'
        elif 'wandb' in name.lower() and current_section != 'wandb':
            lines.append("\n# Weights & Biases logging")
            current_section = 'wandb'
        elif 'dist' in name.lower() and current_section != 'dist':
            lines.append("\n# Distributed training")
            current_section = 'dist'
        
        # Format the config line
        if arg['help']:
            lines.append(f"# {arg['help']}")
        
        default = arg['default']
        if default:
            # Clean up default value
            default = default.replace('"', '').replace("'", '')
            if default == 'None':
                default = 'null'
            elif default in ['True', 'False']:
                default = default.lower()
            elif default.startswith('[') and default.endswith(']'):
                # Keep lists as-is
                pass
            elif arg['type'] in ['int', 'float'] and default.replace('.', '').replace('-', '').isdigit():
                # Numeric value
                pass
            else:
                # String value - quote it
                default = f"'{default}'"
        else:
            default = 'null' if not arg['is_bool'] else 'false'
        
        lines.append(f"{name}: {default}")
    
    # Write to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Generated Hydra config: {output_path}")
    print(f"Total parameters: {len(args_list)}")


def check_conversion_status(script_path: str) -> Dict[str, bool]:
    """Check if a script has been converted to Hydra."""
    with open(script_path, 'r') as f:
        content = f.read()
    
    return {
        'has_argparse': 'import argparse' in content,
        'has_hydra': 'import hydra' in content,
        'has_hydra_main': '@hydra.main' in content,
        'has_get_args': 'def get_args(' in content,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_to_hydra.py <script_path> [output_config_path]")
        print("\nExample:")
        print("  python convert_to_hydra.py run_generation.py conf/config_generation.yaml")
        print("\nOr check status of all scripts:")
        print("  python convert_to_hydra.py --check-all")
        return
    
    if sys.argv[1] == '--check-all':
        scripts = [
            'run_training_vqvae.py',
            'run_training_4m.py',
            'run_generation.py',
            'run_training_divae.py',
            'run_training_vqcontrolnet.py',
            'run_training_4m_fsdp.py',
            'train_wordpiece_tokenizer.py',
            'save_vq_tokens.py',
        ]
        
        print("\n" + "="*80)
        print("HYDRA MIGRATION STATUS")
        print("="*80 + "\n")
        
        for script in scripts:
            if Path(script).exists():
                status = check_conversion_status(script)
                hydra_status = "✅ MIGRATED" if (status['has_hydra'] and not status['has_argparse']) else "🔄 NEEDS MIGRATION"
                print(f"{script:40} {hydra_status}")
                if status['has_argparse'] and status['has_hydra']:
                    print(f"{'':40} ⚠️  Has both argparse and Hydra - partial migration")
            else:
                print(f"{script:40} ❌ NOT FOUND")
        print()
        return
    
    script_path = sys.argv[1]
    output_config = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(script_path).exists():
        print(f"Error: Script not found: {script_path}")
        return
    
    # Check current status
    status = check_conversion_status(script_path)
    print(f"\nScript: {script_path}")
    print(f"  Has argparse: {status['has_argparse']}")
    print(f"  Has Hydra: {status['has_hydra']}")
    print(f"  Has @hydra.main: {status['has_hydra_main']}")
    print(f"  Has get_args(): {status['has_get_args']}")
    
    if not status['has_argparse']:
        print("\n✅ Script appears to already be using Hydra (no argparse found)")
        return
    
    # Extract arguments
    print("\nExtracting argparse arguments...")
    args_list = extract_argparse_args(script_path)
    print(f"Found {len(args_list)} arguments")
    
    # Generate config if output path provided
    if output_config:
        generate_hydra_config(args_list, output_config)
    else:
        print("\nTo generate Hydra config, provide output path:")
        print(f"  python convert_to_hydra.py {script_path} conf/config_name.yaml")


if __name__ == '__main__':
    main()
