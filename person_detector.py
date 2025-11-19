import torch
import torchvision
from torchvision.transforms import functional as F
import cv2
import sys
import os
import time
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import numpy as np

# ----------------------------- 
# LOAD PERSON DETECTOR
# ----------------------------- 
print("Loading person detection model (FasterRCNN)...")
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
model.eval()
print("Person detector loaded.")

# ----------------------------- 
# LOAD HELMET CLASSIFIER
# ----------------------------- 
print("Loading helmet classifier...")
helmet_model_data = torch.load("resnet18_helmet_binary.pth", map_location="cpu")
class_names = helmet_model_data["classes"]  # ['helmet', 'no_helmet']
helmet_model = models.resnet18(weights=None)
num_ftrs = helmet_model.fc.in_features
helmet_model.fc = nn.Linear(num_ftrs, len(class_names))
helmet_model.load_state_dict(helmet_model_data["state_dict"])
helmet_model.eval()

helmet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
])
print("Helmet/no_helmet classifier loaded.")


# ----------------------------- 
# IMPROVED COLOR DETECTION
# ----------------------------- 
def detect_helmet_color(head_crop, debug=False):
    """
    Enhanced color detection for white and yellow helmets with adaptive thresholds.
    Returns: 'white', 'yellow', or None
    """
    h_h, h_w = head_crop.shape[:2]
    
    # Analyze MULTIPLE regions to handle partial occlusion
    regions = [
        (0, int(0.4 * h_h), int(0.2 * h_w), int(0.8 * h_w)),  # Top 40%, wider
        (0, int(0.3 * h_h), int(0.3 * h_w), int(0.7 * h_w)),  # Top 30%, center
        (int(0.1 * h_h), int(0.5 * h_h), int(0.25 * h_w), int(0.75 * h_w)),  # Upper-mid
    ]
    
    color_votes = {'white': 0, 'yellow': 0}
    max_white_score = 0
    max_yellow_score = 0
    
    for y1, y2, x1, x2 in regions:
        helmet_region = head_crop[y1:y2, x1:x2]
        
        if helmet_region.size == 0:
            continue
            
        # Convert to multiple color spaces for robust detection
        hsv = cv2.cvtColor(helmet_region, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(helmet_region, cv2.COLOR_BGR2LAB)
        
        # HSV channels
        h_ch, s_ch, v_ch = cv2.split(hsv)
        mean_h = float(h_ch.mean())
        mean_s = float(s_ch.mean())
        mean_v = float(v_ch.mean())
        std_s = float(s_ch.std())  # Standard deviation for consistency check
        
        # LAB channels (better for white detection)
        l_ch, a_ch, b_ch = cv2.split(lab)
        mean_l = float(l_ch.mean())
        mean_a = float(a_ch.mean())
        mean_b = float(b_ch.mean())
        
        if debug:
            print(f"Region: H={mean_h:.1f}, S={mean_s:.1f}±{std_s:.1f}, V={mean_v:.1f}, "
                  f"L={mean_l:.1f}, a={mean_a:.1f}, b={mean_b:.1f}")
        
        # ===== YELLOW DETECTION (more lenient) =====
        yellow_score = 0
        # Yellow has H around 15-40 in HSV
        if 15 < mean_h < 40 and mean_s > 50 and mean_v > 80:
            yellow_score += 2
        # LAB space: positive b channel (yellowish)
        if mean_b > 135 and mean_l > 100 and mean_a < 140:
            yellow_score += 2
        # Weak yellow signal
        if 12 < mean_h < 45 and mean_s > 40:
            yellow_score += 1
            
        if yellow_score >= 2:
            color_votes['yellow'] += 1
            max_yellow_score = max(max_yellow_score, yellow_score)
        
        # ===== WHITE DETECTION (much more aggressive) =====
        white_score = 0
        
        # Strategy 1: High brightness, low saturation (VERY LENIENT)
        if mean_v > 90 and mean_s < 110:  # Further lowered thresholds
            white_score += 2
        if mean_v > 110 and mean_s < 95:
            white_score += 1
            
        # Strategy 2: LAB space - high lightness, neutral colors (RELAXED)
        if mean_l > 120 and 105 < mean_a < 150 and 105 < mean_b < 155:  # Wider range
            white_score += 2
        if mean_l > 140:  # Very bright
            white_score += 1
            
        # Strategy 3: Pixel distribution analysis (CRITICAL FOR YOUR CASE)
        high_brightness_pixels = np.sum(v_ch > 90) / v_ch.size  # Lowered from 100
        if high_brightness_pixels > 0.35 and mean_s < 120:  # More lenient
            white_score += 2
        if high_brightness_pixels > 0.5:
            white_score += 1
            
        # Strategy 4: Consistent low saturation (uniform white)
        if std_s < 35 and mean_s < 105 and mean_v > 85:  # All relaxed
            white_score += 2
            
        # Strategy 5: RGB analysis - all channels similar and bright
        b, g, r = cv2.split(helmet_region)
        mean_r, mean_g, mean_b_rgb = r.mean(), g.mean(), b.mean()
        rgb_diff = max(mean_r, mean_g, mean_b_rgb) - min(mean_r, mean_g, mean_b_rgb)
        if rgb_diff < 40 and min(mean_r, mean_g, mean_b_rgb) > 90:  # More lenient
            white_score += 2
        if min(mean_r, mean_g, mean_b_rgb) > 120:  # Very bright RGB
            white_score += 1
            
        if white_score >= 3:  # Need at least 3 points to count as white
            color_votes['white'] += 1
            max_white_score = max(max_white_score, white_score)
    
    if debug:
        print(f"Votes - White: {color_votes['white']} (score: {max_white_score}), "
              f"Yellow: {color_votes['yellow']} (score: {max_yellow_score})")
    
    # Decision logic: prioritize strong signals
    if max_yellow_score >= 3 and color_votes['yellow'] >= 1:
        return 'yellow'
    elif max_white_score >= 4 and color_votes['white'] >= 1:
        return 'white'
    elif color_votes['white'] >= 2:  # Two regions agree on white
        return 'white'
    elif color_votes['yellow'] >= 2:  # Two regions agree on yellow
        return 'yellow'
    elif max_white_score >= 5:  # Very strong single white signal
        return 'white'
    elif max_yellow_score >= 4:  # Strong single yellow signal
        return 'yellow'
    
    return None


# ----------------------------- 
# VIDEO INPUT
# ----------------------------- 
video_path = "a9.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: could not open video file: {video_path}")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video props -> FPS: {fps}, Width: {w}, Height: {h}, Frames: {total_frames}")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Changed to mp4v for better compatibility
out = cv2.VideoWriter("w1.mp4", fourcc, fps if fps > 0 else 25.0, (w, h))

os.makedirs("head_crops", exist_ok=True)

# ===== REAL-TIME DISPLAY OPTION =====
SHOW_REALTIME = True  # Set to False if you don't want to see video while processing
if SHOW_REALTIME:
    print("\n** REAL-TIME DISPLAY ENABLED **")
    print("Press 'q' to quit early, 'p' to pause/resume, SPACE to skip frame\n")
    
paused = False

# ----------------------------- 
# PROCESS VIDEO
# ----------------------------- 
frame_idx = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("No more frames – exiting.")
        break
    
    frame_idx += 1
    
    # Progress log
    if frame_idx % 10 == 0 and total_frames > 0:
        elapsed = time.time() - start_time
        fps_now = frame_idx / elapsed if elapsed > 0 else 0
        percent = (frame_idx / total_frames) * 100
        eta = (total_frames - frame_idx) / fps_now if fps_now > 0 else 0
        print(f"Frame {frame_idx}/{total_frames} "
              f"({percent:.2f}%) | FPS: {fps_now:.2f} | ETA: {eta:.1f}s")
    
    # ===== PAUSE FUNCTIONALITY =====
    if SHOW_REALTIME and paused:
        cv2.imshow("Helmet Detection (PAUSED - Press 'p' to resume)", frame)
        key = cv2.waitKey(100) & 0xFF
        if key == ord('p'):
            paused = False
            print("Resumed...")
        elif key == ord('q'):
            print("Quitting...")
            break
        continue
    
    # Person detection
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = F.to_tensor(img_rgb).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(img_tensor)[0]
    
    boxes = outputs["boxes"]
    labels = outputs["labels"]
    scores = outputs["scores"]
    
    person_id = 0
    
    for box, label, score in zip(boxes, labels, scores):
        if label.item() == 1 and score.item() > 0.7:  # person class
            x1, y1, x2, y2 = box.int().tolist()
            
            # PERSON BOX (green)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # HEAD CROP - slightly larger region
            person_w = x2 - x1
            person_h = y2 - y1
            helmet_h = int(0.65 * person_w)  # Increased from 0.6
            head_y1 = max(y1, 0)
            head_y2 = min(head_y1 + helmet_h, y2, h - 1)
            
            if head_y2 <= head_y1 + 5:  # Increased minimum height
                continue
            
            # HEAD BOX (blue)
            cv2.rectangle(frame, (x1, head_y1), (x2, head_y2), (255, 0, 0), 2)
            
            head_crop = frame[head_y1:head_y2, x1:x2]
            
            if head_crop.size == 0:
                continue
            
            # Save head crop
            person_id += 1
            crop_filename = f"head_crops/f{frame_idx:05d}_p{person_id}.jpg"
            cv2.imwrite(crop_filename, head_crop)
            
            # ========================================
            # HELMET CLASSIFICATION (CNN)
            # ========================================
            head_rgb = cv2.cvtColor(head_crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(head_rgb)
            input_tensor = helmet_transform(pil_img).unsqueeze(0)
            
            with torch.no_grad():
                logits = helmet_model(input_tensor)
                probs = torch.softmax(logits, dim=1)
                pred_idx = logits.argmax(1).item()
                confidence = probs[0][pred_idx].item()
            
            helmet_label = class_names[pred_idx]  # 'helmet' or 'no_helmet'
            
            # ========================================
            # COLOR DETECTION (ALWAYS RUN - can override CNN)
            # ========================================
            helmet_color = detect_helmet_color(head_crop, debug=False)
            
            # ========================================
            # FINAL LABEL (Color detection overrides CNN)
            # ========================================
            # If we detect white or yellow color, it MUST be a helmet
            if helmet_color == "white":
                final_label = f"white_helmet ({confidence:.2f})"
                label_color = (255, 255, 255)  # White text
            elif helmet_color == "yellow":
                final_label = f"yellow_helmet ({confidence:.2f})"
                label_color = (0, 255, 255)  # Yellow text
            elif helmet_label == "helmet":
                # CNN says helmet but no specific color detected
                final_label = f"helmet ({confidence:.2f})"
                label_color = (0, 255, 0)  # Green text
            else:
                # CNN says no helmet AND no color detected
                final_label = f"no_helmet ({confidence:.2f})"
                label_color = (0, 0, 255)  # Red text
            
            # Draw label with background for readability
            text_size = cv2.getTextSize(final_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, head_y1 - 25), (x1 + text_size[0] + 5, head_y1), (0, 0, 0), -1)
            cv2.putText(
                frame,
                final_label,
                (x1, head_y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                label_color,
                2
            )
    
    out.write(frame)
    
    # ===== REAL-TIME DISPLAY =====
    if SHOW_REALTIME:
        # Resize if frame is too large for screen
        display_frame = frame.copy()
        display_h, display_w = display_frame.shape[:2]
        max_display_width = 1280
        if display_w > max_display_width:
            scale = max_display_width / display_w
            display_frame = cv2.resize(display_frame, None, fx=scale, fy=scale)
        
        cv2.imshow("Helmet Detection (Press 'q' to quit, 'p' to pause)", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("User quit early.")
            break
        elif key == ord('p'):
            paused = True
            print("Paused. Press 'p' again to resume...")
        elif key == ord(' '):  # Space bar - skip frame
            pass

cap.release()
out.release()
if SHOW_REALTIME:
    cv2.destroyAllWindows()
print("DONE. Saved: w1.mp4")
print(f"Head crops saved in: head_crops/")