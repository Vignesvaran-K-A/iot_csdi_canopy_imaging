import os
import sys
import torch
import cv2
import numpy as np
import torchvision.ops as ops
import torchvision.transforms.functional as F
from PIL import Image
from model import get_instance_segmentation_model

# 7 Target Classes
CLASS_NAMES = ['background', 'leaf', 'flower', 'bud', 'pre_mature_pod', 'mature_pod', 'post_mature_pod']

# Distinct BGR colors for OpenCV
COLORS = [
    (0, 0, 0),         # background
    (0, 220, 0),       # leaf (Vibrant Green)
    (255, 0, 255),     # flower (Magenta)
    (0, 255, 255),     # bud (Yellow)
    (255, 140, 0),     # pre_mature_pod (Orange/Cyan)
    (0, 0, 255),       # mature_pod (Red)
    (180, 0, 180)      # post_mature_pod (Purple)
]

# Class-adaptive thresholds:
# Leaves dominate the dataset (~70% of annotations) and cause background false positives,
# so we apply a higher threshold (0.70) for leaves while keeping high sensitivity (0.50)
# for reproductive organs (flowers, buds, pods).
CLASS_THRESHOLDS = {
    'leaf': 0.70,
    'flower': 0.50,
    'bud': 0.50,
    'pre_mature_pod': 0.50,
    'mature_pod': 0.50,
    'post_mature_pod': 0.50
}

def resolve_path(relative_path):
    """Finds file relative to current directory or script directory."""
    if os.path.exists(relative_path):
        return relative_path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate1 = os.path.join(script_dir, relative_path)
    if os.path.exists(candidate1):
        return candidate1
    candidate2 = os.path.join(script_dir, "..", relative_path)
    if os.path.exists(candidate2):
        return candidate2
    return relative_path

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Running inference on: {device}")

    # 1. Locate weights
    weights_path = resolve_path("canopy_maskrcnn.pth")
    if not os.path.exists(weights_path):
        print(f"Error: Could not find weights at {weights_path}")
        return

    # Load Model
    model = get_instance_segmentation_model(num_classes=7)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # 2. Select Test Image (via CLI argument or default sample)
    if len(sys.argv) > 1:
        test_img_path = sys.argv[1]
    else:
        test_img_path = r"dataset_processed\day_07\hsie_enhanced\canopy_20260930_122702_hsie.png"

    resolved_img_path = resolve_path(test_img_path)
    if not os.path.exists(resolved_img_path):
        # Fallback to any available HSIE test image
        fallback = resolve_path(r"dataset_processed\day_02\hsie_enhanced\canopy_20260828_150037_hsie.png")
        if os.path.exists(fallback):
            resolved_img_path = fallback
        else:
            print(f"Error: Could not find test image at {resolved_img_path}")
            return

    print(f"Processing image: {resolved_img_path}")
    img_pil = Image.open(resolved_img_path).convert("RGB")
    target_size = (1024, 768)
    img_pil_resized = img_pil.resize(target_size, Image.BILINEAR)
    img_tensor = F.to_tensor(img_pil_resized).unsqueeze(0).to(device)

    # 3. Model Inference
    with torch.no_grad():
        predictions = model(img_tensor)[0]

    # 4. Class-Adaptive Filtering
    scores = predictions['scores']
    labels = predictions['labels']
    boxes = predictions['boxes']
    masks = predictions['masks']

    keep_indices = []
    for i in range(len(scores)):
        score = scores[i].item()
        lbl_id = labels[i].item()
        lbl_name = CLASS_NAMES[lbl_id]
        thresh = CLASS_THRESHOLDS.get(lbl_name, 0.50)
        if score >= thresh:
            keep_indices.append(i)

    if len(keep_indices) == 0:
        print("No instances found above threshold.")
        return

    filtered_boxes = boxes[keep_indices]
    filtered_scores = scores[keep_indices]
    filtered_labels = labels[keep_indices]
    filtered_masks = masks[keep_indices]

    # 5. CROSS-CLASS NON-MAXIMUM SUPPRESSION (Class-Agnostic NMS)
    # This prevents the exact same region from being labeled as BOTH a flower and a leaf!
    # If two boxes overlap with IoU > 0.35, the lower-confidence class is suppressed.
    nms_keep = ops.nms(filtered_boxes, filtered_scores, iou_threshold=0.35)

    final_boxes = filtered_boxes[nms_keep].cpu().numpy().astype(int)
    final_scores = filtered_scores[nms_keep].cpu().numpy()
    final_labels = filtered_labels[nms_keep].cpu().numpy()
    final_masks = filtered_masks[nms_keep, 0].mul(255).byte().cpu().numpy()

    # 6. Render Overlays onto image
    img_cv = cv2.cvtColor(np.array(img_pil_resized), cv2.COLOR_RGB2BGR)
    overlay = img_cv.copy()

    instances_found = len(final_boxes)
    print(f"\nFinal Confirmed Detections ({instances_found} instances):")

    for i in range(instances_found):
        box = final_boxes[i]
        score = final_scores[i]
        label_id = final_labels[i]
        label_name = CLASS_NAMES[label_id]
        color = COLORS[label_id]
        mask = final_masks[i]

        print(f"  [{i+1}] {label_name:16s} - Confidence: {score:.2f} - Box: {box.tolist()}")

        # Fill mask on overlay
        overlay[mask > 127] = color

        # Draw contour line around mask boundary for sharp edges
        contours, _ = cv2.findContours((mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_cv, contours, -1, color, 2)

        # Draw Bounding Box
        cv2.rectangle(img_cv, (box[0], box[1]), (box[2], box[3]), color, 2)

        # Draw Label Banner with solid background for readability
        label_text = f"{label_name}: {score:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_text = max(box[1], text_h + 5)
        cv2.rectangle(img_cv, (box[0], y_text - text_h - 4), (box[0] + text_w + 4, y_text + baseline - 2), color, -1)
        cv2.putText(img_cv, label_text, (box[0] + 2, y_text - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Blend mask overlay with original image (alpha 0.40)
    cv2.addWeighted(overlay, 0.40, img_cv, 0.60, 0, img_cv)

    # 7. Save Result
    output_path = resolve_path("pilot_prediction_result.png")
    cv2.imwrite(output_path, img_cv)
    print(f"\nSuccessfully saved prediction visualization to: {output_path}")

if __name__ == "__main__":
    main()
