import torch
import torchvision.io as io
from torchvision.transforms import functional as F
from omegaconf import OmegaConf
import time
from tracker.basic_tracker import Tracker

def check_tracker(video_path: str):
    print(f"Loading video from: {video_path}")
    
    # 1. Load video frames
    # read_video returns (vframes [T, H, W, C], aframes, info)
    try:
        vframes, _, _ = io.read_video(video_path, pts_unit='sec')
    except Exception as e:
        print(f"Failed to load video. Ensure torchvision and av are installed. Error: {e}")
        return

    # 2. Format to match your class: (B, T, C, H, W)
    # Convert [T, H, W, C] to [T, C, H, W]
    vframes = vframes.permute(0, 3, 1, 2)
    
    # Optional: Resize to 640x640 (RTDETR standard) to prevent OOM on the cluster
    vframes = F.resize(vframes, [640, 640])
    
    # Add batch dimension -> (1, T, C, H, W)
    x = vframes.unsqueeze(0)
    
    # Ultralytics models usually expect float tensors normalized to [0, 1]
    if x.dtype == torch.uint8:
        x = x.float() / 255.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    print(f"Video prepared. Input shape: {x.shape} on {device}")

    # 3. Mock Configuration
    cfg = OmegaConf.create({
        "max_track_tokens": 10,
        "hpatches": 16,
        "wpatches": 16,
        "embedding_dim": 256,
        "num_frames": 64,                 # Target output frames
        "track_model": "rtdetr-l.pt"     # Will auto-download if missing
    })

    print("\nInitializing Tracker...")
    tracker = Tracker(cfg).to(device)
    tracker.eval() # Run in eval mode for testing

    print("\nRunning forward pass (Tracking)...")
    start_time = time.time()
    
    with torch.no_grad():
        out, all_tracks, all_frames = tracker(x)
        
    end_time = time.time()
    
    # 4. Diagnostics and Validation
    print(f"\n--- Tracking completed in {end_time - start_time:.2f} seconds ---")
    print(f"Total frames in source video: {x.shape[1]}")
    print(f"Selected frame indices:       {all_frames[0]}")
    print(f"Output embeddings shape:      {out.shape}")
    print(f"Output tracks shape:          {all_tracks.shape}")

    # 5. Assertions to guarantee correctness
    try:
        assert out.shape == (1, cfg.num_frames, cfg.max_track_tokens, cfg.embedding_dim), "Embedding shape mismatch!"
        assert all_tracks.shape == (1, cfg.num_frames, cfg.max_track_tokens), "Tracks tensor shape mismatch!"
        assert len(all_frames[0]) == cfg.num_frames, f"Expected {cfg.num_frames} frames, but got {len(all_frames[0])}!"
        print("\n✅ All structural checks passed! The tracker logic is healthy.")
        
        # Quick visual check of the temporal split
        t_split = x.shape[1] // 3
        late_frames = [f for f in all_frames[0] if f >= t_split]
        print(f"✅ {len(late_frames)} out of {cfg.num_frames} selected frames came from the last 2/3 of the video.")
        
    except AssertionError as e:
        print(f"\n❌ Assertion Failed: {e}")

if __name__ == "__main__":
    test_video = "/sc/home/konrad.goldenbaum/data/SoccerData/mvfouls/train/action_0/clip_0.mp4"
    check_tracker(test_video)
