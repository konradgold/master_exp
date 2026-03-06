from ultralytics import RTDETR
import torch

try:
    model = RTDETR("rtdetr-l.pt")
    print("RTDETR loaded.")
    if hasattr(model, "fuse"):
        print("Model has fuse() method.")
    else:
        print("Model does NOT have fuse() method.")
        
    # Check if model.model has fuse
    if hasattr(model.model, "fuse"):
        print("Inner model has fuse() method.")
        
except Exception as e:
    print(f"Error: {e}")
