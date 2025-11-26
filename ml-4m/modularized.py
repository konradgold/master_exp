from tabnanny import check
import hydra
from regex import P
from fourm.models.fm import FM
from pathlib import Path
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from fourm.utils.checkpoint import load_safetensors

def align_config(cfg, checkpoint_config):
    # Implement your validation logic here
    # For example, compare cfg and checkpoint_config attributes
    checkpoint_config['domains_in'] = cfg.domains_in
    checkpoint_config['domains_out'] = cfg.domains_out
    return checkpoint_config

@hydra.main(version_base=None, config_path="cfgs", config_name="default_run")
def main(cfg):
    print(cfg.run_name)
    # Check if checkpoint exists locally
    checkpoint_path = Path(cfg.checkpoint_path) if hasattr(cfg, 'checkpoint_path') else None
        
    if checkpoint_path and checkpoint_path.exists():
        print(f"Loading checkpoint from local path: {checkpoint_path}")
        filename = Path(cfg.checkpoint_path) / "model.safetensors"
    else:
        print(f"Checkpoint not found locally, loading from HuggingFace: {cfg.backbone}")
        # Create directory if it doesn't exist
        checkpoint_path = Path(cfg.checkpoint_path) / "model.safetensors"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download checkpoint from HuggingFace
        filename = hf_hub_download(
            repo_id=cfg.backbone,
            filename="model.safetensors",
            local_dir=checkpoint_path.parent,
            local_dir_use_symlinks=False
        )
    # Load the checkpoint
    state_dict, config = load_safetensors(str(filename))
    config = align_config(cfg, config)
    model = FM(config)
    model.load_state_dict(state_dict, strict=False)
    print("model loaded successfully")

    

main()