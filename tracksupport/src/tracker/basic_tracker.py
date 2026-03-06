import torch.nn as nn
import torch
from omegaconf import DictConfig
from ultralytics import RTDETR

class Tracker(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.max_track_tokens = cfg.max_track_tokens
        self.hpatches = cfg.hpatches
        self.wpatches = cfg.wpatches
        self.num_patches = self.hpatches * self.wpatches
        self.embedding_dim = cfg.embedding_dim
        self.num_frames = cfg.num_frames
        self.positional_embedding = nn.Parameter(torch.randn(self.num_patches, self.embedding_dim))
        self.tracker: RTDETR = RTDETR(cfg.track_model)
        self.tracker.overrides["verbose"] = False

    def train(self, mode: bool = True):
        # Skip calling .train() on the Ultralytics RTDETR model,
        # which overrides .train() to launch a training procedure.
        self.training = mode
        for name, module in self._modules.items():
            if name == "tracker":
                continue
            module.train(mode)
        return self

    def forward(self, x: torch.Tensor):
        B, T, H, W, C = x.shape
        out = torch.zeros(B, self.num_frames, self.max_track_tokens, self.embedding_dim, device=x.device)
        all_tracks = []
        all_frames = []
        for b in range(B):
            tracks, frames = self._find_tracks(x[b])
            all_frames.append(frames)
            # Contract: at track t, the number is either the patch index (0 to num_patches-1, t = i*self.hpatches + j) or -1 for padding
            if tracks.ndim == 2:
                tracks = tracks.unsqueeze(0)  # Add batch dimension if missing
            all_tracks.append(tracks)

            valid_mask = tracks >= 0
            valid_tracks = tracks[valid_mask]
            assert valid_tracks.max() < self.num_patches, f"Track index {valid_tracks.max()} is out of bounds for {self.num_patches} patches"
            out[b][valid_mask.squeeze()] = self.positional_embedding[valid_tracks.long()]
            
        return out, torch.cat(all_tracks, dim=0), all_frames

    def _find_tracks(self, x: torch.Tensor):
        T, C, H, W = x.shape
        if hasattr(self.tracker, 'predictor') and hasattr(self.tracker.predictor, 'trackers'):
            for t in self.tracker.predictor.trackers:
                t.reset()
        results = self.tracker.track(
                source=x,
                tracker="bytetrack.yaml",
                persist=True,
                stream=True,
                verbose=False,
            )
        
        track_lifetimes = {}
        frame_detections = {t: [] for t in range(T)}

        for frame_idx, r in enumerate(results):
            boxes = r.boxes
            if boxes.id is None:
                continue

            xywhn = boxes.xywhn.cpu().tolist()

            ids = boxes.id.int().cpu().tolist()

            for tid, (cxn, cyn, w_n, h_n) in zip(ids, xywhn):
                tid = int(tid)


                # normalized center directly
                w_id = int(cxn*self.wpatches) if int(cxn*self.wpatches) < self.wpatches else self.wpatches - 1
                assert w_id < self.wpatches, f"Calculated patch index {w_id} is out of bounds for {self.wpatches} patches"
                h_id = int(cyn*self.hpatches) if int(cyn*self.hpatches) < self.hpatches else self.hpatches - 1
                assert h_id < self.hpatches, f"Calculated patch index {h_id} is out of bounds for {self.hpatches} patches"

                pid = h_id * self.wpatches + w_id

                frame_detections[frame_idx].append((tid, pid))
                track_lifetimes[tid] = track_lifetimes.get(tid, 0) + 1

        longest_tids = sorted(track_lifetimes.keys(), key=lambda t: track_lifetimes[t], reverse=True)

        top_tids = longest_tids[:self.max_track_tokens]
        tid_to_idx = {tid: idx for idx, tid in enumerate(top_tids)}

        valid_frames = [f for f in range(T) if any(t[0] in tid_to_idx for t in frame_detections[f])]

        frames = self._select_frames(valid_frames, self.num_frames, len(valid_frames), T)

        result_tensor = torch.full((self.num_frames, self.max_track_tokens), -1, dtype=torch.long, device=x.device)

        for out_f_idx, f_idx in enumerate(frames):
            for tid, pid in frame_detections[f_idx]:
                if tid in tid_to_idx:
                    slot_idx = tid_to_idx[tid]
                    result_tensor[out_f_idx, slot_idx] = pid
        
        assert result_tensor.shape == (self.num_frames, self.max_track_tokens), f"Expected result tensor shape {(self.num_frames, self.max_track_tokens)}, got {result_tensor.shape}"
        assert len(frames) == self.num_frames, f"Expected {self.num_frames} frames, got {len(frames)}"


        return result_tensor, frames

    
    def _select_frames(self, valid_frames, N, V, T):
        if V == 0:
            # Fallback: no tracks found at all, sample evenly across the whole video
            frames = torch.linspace(0, T - 1, N).long().tolist()

        elif V > N:
            # We have MORE valid frames than needed. Time to downsample.
            t_split = T // 3
            early_pool = [f for f in valid_frames if f < t_split]
            late_pool = [f for f in valid_frames if f >= t_split]
            
            if len(late_pool) >= N:
                # 100% of frames come from the last 2/3 of the video
                indices = torch.linspace(0, len(late_pool) - 1, N).long()
                frames = [late_pool[i] for i in indices]
            else:
                # Keep all available late frames, pad the rest from the early pool
                missing = N - len(late_pool)
                indices = torch.linspace(0, len(early_pool) - 1, missing).long()
                early_frames = [early_pool[i] for i in indices]
                
                # early_frames are already smaller than late_pool frames, so appending maintains sorted order
                frames = early_frames + late_pool 
        else:
            # V <= N: We don't have enough tracked frames. Keep them all and pad.
            frames = valid_frames
            missing = N - V
            if missing > 0:
                available = [f for f in range(T) if f not in frames]
                # Pad by sampling evenly from the remaining untracked frames
                pad_indices = torch.linspace(0, len(available) - 1, missing).long()
                frames += [available[i] for i in pad_indices]
                frames = sorted(frames)
        return frames


