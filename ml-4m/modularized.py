import hydra
from fourm.models.fm import FM
from run_generation import load_model

@hydra.main(version_base=None, config_path="cfgs", config_name="default_run")
def main(cfg):
    print(cfg.run_name)
    model = FM
    model = load_model(cfg.model, FM, cfg.device)

main()