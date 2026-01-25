import argparse
import torch
import cv2
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification
from facenet_pytorch import MTCNN
from collections import deque
from tqdm import tqdm
from PIL import Image
import os

def get_model(device="cpu"):
    print("⬇️  Loading ViT Model...")
    model_id = "dima806/deepfake_vs_real_image_detection"
    try:
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForImageClassification.from_pretrained(model_id)
    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)
    model.to(device)
    model.eval()
    return model, processor

def draw_hud(frame, box, score, threshold):
    """Draws the Red Box and Score on the frame"""
    if box is None: return frame
    
    # Color: Red if high score (FAKE), Green if low (REAL)
    color = (0, 0, 255) if score > threshold else (0, 255, 0)
    
    # Draw Box
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Draw Text Background
    label = f"FAKE: {score*100:.1f}%"
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
    
    # Draw Text
    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame

def process_video(video_path, stride=5, threshold=0.6):
    device = "cpu"
    print(f"🚀 Initializing Visualizer on {device.upper()}...")

    mtcnn = MTCNN(image_size=224, margin=10, keep_all=False, select_largest=True, device=device)
    model, processor = get_model(device)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"🎬 Processing: {video_path}")
    
    half_window = int(fps * 1.5) 
    frame_buffer = deque(maxlen=half_window)
    
    candidates = [] 
    clip_counter = 0
    
    fake_id = 1
    for idx, label in model.config.id2label.items():
        if "fake" in label.lower(): fake_id = idx

    frame_idx = 0
    is_recording = False
    frames_to_record = 0
    writer = None
    current_clip_max = 0.0
    current_clip_file = ""
    
    # State persistence
    last_box = None
    last_score = 0.0

    pbar = tqdm(total=total_frames)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # --- DETECTION ---
        run_ai = (frame_idx % stride == 0) or is_recording
        
        # We start with the last known state
        box_to_draw = last_box
        score_to_draw = last_score

        if run_ai:
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                
                # We use detect() to get raw boxes
                boxes, _ = mtcnn.detect(pil_img)
                
                if boxes is not None:
                    box = boxes[0].astype(int)
                    # Safe Crop limits
                    box[0]=max(0,box[0]); box[1]=max(0,box[1])
                    box[2]=min(width,box[2]); box[3]=min(height,box[3])
                    
                    face_crop = pil_img.crop((box[0], box[1], box[2], box[3]))
                    inputs = processor(images=face_crop, return_tensors="pt").to(device)
                    with torch.no_grad():
                        out = model(**inputs)
                        current_score = out.logits.softmax(dim=-1)[0][fake_id].item()
                    
                    # Update State
                    last_box = box
                    last_score = current_score
                    
                    box_to_draw = box
                    score_to_draw = current_score
            except:
                pass

        # --- TRIGGER LOGIC ---
        if score_to_draw > threshold and not is_recording:
            is_recording = True
            frames_to_record = half_window # Record 1.5s future
            clip_counter += 1
            current_clip_max = score_to_draw
            current_clip_file = f"temp_vis_{clip_counter}.mp4"
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(current_clip_file, fourcc, fps, (width, height))
            
            # Dump Buffer (Clean History)
            # We don't draw boxes on history frames to save processing time
            while frame_buffer:
                writer.write(frame_buffer.popleft())
                
            pbar.write(f"🎥 Triggered! Recording Clip #{clip_counter}")

        # --- BUFFER & RECORD ---
        
        # If we are NOT recording, buffer the clean frame
        if not is_recording:
            frame_buffer.append(frame.copy())
            
        # If we ARE recording
        if is_recording:
            if score_to_draw > current_clip_max: current_clip_max = score_to_draw
            
            # Draw HUD on the frame
            vis_frame = frame.copy()
            vis_frame = draw_hud(vis_frame, box_to_draw, score_to_draw, threshold)
            
            writer.write(vis_frame)
            frames_to_record -= 1
            
            if frames_to_record <= 0:
                writer.release()
                is_recording = False
                writer = None
                
                candidates.append({'file': current_clip_file, 'score': current_clip_max})
                frame_buffer.clear()
                last_box = None 

        pbar.update(1)
        frame_idx += 1

    cap.release()
    if writer: writer.release()
    pbar.close()

    # --- FILTER TOP 3 ---
    print("\n" + "="*50)
    print("🧹 FILTERING: Keeping Top 3 Visualized Clips")
    
    if not candidates:
        print("🤷 No clips found.")
        return

    candidates.sort(key=lambda x: x['score'], reverse=True)
    
    # Clean up files
    for i, item in enumerate(candidates):
        old_name = item['file']
        if i < 3:
            new_name = f"VISUAL_EVIDENCE_{i+1}_score_{item['score']:.2f}.mp4"
            if os.path.exists(new_name): os.remove(new_name)
            os.rename(old_name, new_name)
            print(f"🏆 SAVED: {new_name} (Peak: {item['score']*100:.1f}%)")
        else:
            if os.path.exists(old_name):
                os.remove(old_name)
                
    print(f"🗑️  Cleaned up {len(candidates) - min(3, len(candidates))} unused clips.")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.6)
    args = parser.parse_args()
    
    process_video(args.input, args.stride, args.threshold)
