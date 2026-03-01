import hydra
from omegaconf import DictConfig
from torch import nn
import torch
import warnings

from src.tracker.basic_tracker import Tracker
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    from torchvision.io.video import read_video
from transformers import AutoModel, AutoVideoProcessor
from src.classification.attention_pooling import AttentivePooler
from src.classification.mvfoulhead import EmbedMVAggregate
from src.tracker.modules import TrackTention

class VJEPATracker(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.tracker = Tracker(cfg)
        self.tracktention = TrackTention(cfg)
        self.processor = AutoVideoProcessor.from_pretrained(cfg.model_hf)
        self.embedding = AutoModel.from_pretrained(cfg.model_hf)
        self.num_frames = cfg.num_frames
        self.wpatches = cfg.wpatches
        self.hpatches = cfg.hpatches
        self.max_track_tokens = cfg.max_track_tokens

        self.aggregation = AttentivePooler(cfg.modules)
        self.classifier = EmbedMVAggregate(agr_type=cfg.agr_type, feat_dim=cfg.embedding_dim, return_attention=cfg.return_attention)
    
    def train(self, mode: bool = True):
        # Skip calling .train() on the Ultralytics RTDETR model,
        # which overrides .train() to launch a training procedure.
        self.training = mode
        for name, module in self._modules.items():
            if name == "processor" or name == "embedding":
                continue
            module.train(mode)
        return self

    def forward(self, x):
        b, v, t, c, h, w = x.shape
        x = x.view(b*v, t, c, h, w)
        x_list = [x[i] for i in range(x.shape[0])]
        vframes = self.processor(videos=x_list, return_tensors="pt")['pixel_values_videos']
        assert vframes.shape[:3] == (b*v, t, c), f"Expected shape {(b*v, t, c, h, w)}, got {vframes.shape}"

        vframes_unscaled = (vframes-vframes.min())/(vframes.max()-vframes.min())
        track_tokens, tracks, frames = self.tracker(vframes_unscaled)
        frames = torch.Tensor(frames).to(vframes.device).long()

        vframes = torch.gather(input=vframes, dim=1, index=frames.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, *vframes.shape[2:]))

        embedded = self.embedding(vframes)['last_hidden_state']
        embedded = embedded.view(b*v, self.num_frames//2, self.wpatches, self.hpatches, -1)
        track_tokens = track_tokens[:,::2].view(b*v, self.num_frames//2, self.max_track_tokens, -1)
        tracks: torch.Tensor = tracks[:,::2]

        track_tokens, tokens = self.tracktention(track_tokens, tracks, embedded)
        aggregated = self.aggregation(tokens.view(b, -1, tokens.shape[-1]))
        pred_action, pred_offence_severity = self.classifier(aggregated)
        assert pred_action.shape[0] == b, f"Expected batch size {b} for pred_action, got {pred_action.shape[0]}"
        assert pred_offence_severity.shape[0] == b, f"Expected batch size {b} for pred_offence_severity, got {pred_offence_severity.shape[0]}"
        return pred_action, pred_offence_severity, track_tokens



@hydra.main(version_base=None, config_path="../cfg", config_name="first")
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VJEPATracker(cfg)
    model.to(device)
    vid_path = "/sc/home/konrad.goldenbaum/data/SoccerData/mvfouls/train/action_43/clip_1.mp4"
    frames = read_video(vid_path, start_pts=0, end_pts=None, pts_unit="sec", output_format="TCHW")[0]
    frames = frames.unsqueeze(0)
    frames = torch.cat([frames, frames], dim=0)
    frames = frames.unsqueeze(0)
    frames = frames.to(device)

    pred_action, pred_offence_severity, track_tokens = model(frames)
    print("Pred action shape:", pred_action.shape)
    print("Pred offence severity shape:", pred_offence_severity.shape)
    print("Track tokens shape:", track_tokens.shape)

if __name__ == "__main__":
    main()
