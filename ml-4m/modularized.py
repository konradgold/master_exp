import hydra
from fourm.models.fm import FM
from pathlib import Path
from huggingface_hub import hf_hub_download

@hydra.main(version_base=None, config_path="cfgs", config_name="default_run")
def main(cfg):
    print(cfg.run_name)
    # Check if checkpoint exists locally
    checkpoint_path = Path(cfg.checkpoint_path) if hasattr(cfg, 'checkpoint_path') else None
        
    if checkpoint_path and checkpoint_path.exists():
        print(f"Loading checkpoint from local path: {checkpoint_path}")
    else:
        print(f"Checkpoint not found locally, loading from HuggingFace: {cfg.backbone}")
        # Create directory if it doesn't exist
        checkpoint_path = Path(cfg.checkpoint_path) / "model.safetensors"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download checkpoint from HuggingFace
        _ = hf_hub_download(
            repo_id=cfg.backbone,
            filename="model.safetensors",
            local_dir=checkpoint_path.parent,
            local_dir_use_symlinks=False
        )

main()