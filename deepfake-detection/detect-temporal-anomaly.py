import torch
import torchvision.models.video as models
import torchvision.transforms.functional as F
import cv2
import numpy as np
import argparse
from tqdm import tqdm

def get_temporal_model():
    print("⬇️  Loading R3D-18 (Temporal Flow Model)...")
    # Uses Kinetics-400 pre-trained weights (Action Recognition)
    model = models.r3d_18(weights=models.R3D_18_Weights.DEFAULT)
    model.eval()
    return model

def process_video(video_path):
    model = get_temporal_model()
    cap = cv2.VideoCapture(video_path)
    
    # R3D expects 16-frame clips
    clip_len = 16 
    buffer = []
    
    print(f"🎬 Scanning Temporal Flow: {video_path}")
    
    frame_idx = 0
    scores = []
    
    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Resize/Norm for model
            frame = cv2.resize(frame, (112, 112))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = F.to_tensor(frame) # [3, 112, 112]
            
            buffer.append(frame_tensor)
            
            if len(buffer) == clip_len:
                # Stack to [1, 3, 16, 112, 112] (Batch, C, T, H, W)
                input_clip = torch.stack(buffer).permute(1, 0, 2, 3).unsqueeze(0)
                
                # Get Feature Vector (The "Meaning" of the motion)
                features = model(input_clip)
                
                # We calculate the "energy" of the motion.
                # Sudden drops/spikes in motion energy indicate splicing.
                score = torch.norm(features).item()
                scores.append((frame_idx, score))
                
                buffer.pop(0) # Sliding window
                
            frame_idx += 1
            if frame_idx % 100 == 0: print(f"   Scanned {frame_idx} frames...")

    cap.release()
    
    # Analyze Scores for Jumps
    print("\n" + "="*50)
    print("📉 TEMPORAL FLOW REPORT")
    print("="*50)
    
    vals = [s[1] for s in scores]
    avg = np.mean(vals)
    std = np.std(vals)
    
    for idx, score in scores:
        # If motion energy changes by 4 standard deviations, it's unnatural
        if abs(score - avg) > (4 * std):
            timestamp = idx / 30.0 # Assuming 30fps
            print(f"✂️  ANOMALY at {int(timestamp//60)}:{int(timestamp%60):02d} | Score: {score:.2f} (Possible Splice)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    process_video(args.input)

