import os
import glob
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
from model import get_instance_segmentation_model
from dataset import CanopyDataset

def box_iou(b1, b2):
    x1 = torch.max(b1[:, None, 0], b2[None, :, 0])
    y1 = torch.max(b1[:, None, 1], b2[None, :, 1])
    x2 = torch.min(b1[:, None, 2], b2[None, :, 2])
    y2 = torch.min(b1[:, None, 3], b2[None, :, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-6)

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Running evaluation on: {device}")

    weights_path = r"D:\canopy_project\canopy_maskrcnn.pth"
    if not os.path.exists(weights_path):
        print(f"Error: {weights_path} not found.")
        return

    model = get_instance_segmentation_model(num_classes=7)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    xml_files = glob.glob(r"D:\canopy_project\*.xml")
    dataset = CanopyDataset(xml_source=xml_files, image_root_dir=r"D:\canopy_project", target_size=(1024, 768))
    print(f"Total dataset images: {len(dataset)}")

    class_names = ['background', 'leaf', 'flower', 'bud', 'pre_mature_pod', 'mature_pod', 'post_mature_pod']
    
    total_gt = {i: 0 for i in range(1, 7)}
    tp_count = {i: 0 for i in range(1, 7)}
    fp_count = {i: 0 for i in range(1, 7)}
    
    flower_leaf_collisions = 0
    flower_count_with_preds = 0
    confusion = np.zeros((7, 7), dtype=int)

    with torch.no_grad():
        for idx in range(len(dataset)):
            img_pil, target = dataset[idx]
            gt_boxes = target['boxes'].to(device)
            gt_labels = target['labels'].to(device)

            for lbl in gt_labels:
                total_gt[lbl.item()] += 1

            img_t = F.to_tensor(img_pil).unsqueeze(0).to(device)
            preds = model(img_t)[0]

            # Filter with threshold 0.5 (same as predict.py)
            conf_mask = preds['scores'] >= 0.5
            pred_boxes = preds['boxes'][conf_mask]
            pred_labels = preds['labels'][conf_mask]

            # Check cross-class flower/leaf collision
            f_idx = (pred_labels == 2).nonzero(as_tuple=True)[0]
            l_idx = (pred_labels == 1).nonzero(as_tuple=True)[0]
            if len(f_idx) > 0:
                flower_count_with_preds += 1
                if len(l_idx) > 0:
                    f_ious = box_iou(pred_boxes[f_idx], pred_boxes[l_idx])
                    if (f_ious > 0.35).any():
                        flower_leaf_collisions += 1

            if len(gt_boxes) == 0:
                for pl in pred_labels:
                    fp_count[pl.item()] += 1
                continue

            if len(pred_boxes) == 0:
                for gl in gt_labels:
                    confusion[gl.item(), 0] += 1
                continue

            ious = box_iou(gt_boxes, pred_boxes)
            matched_preds = set()
            for g_i in range(len(gt_boxes)):
                g_lbl = gt_labels[g_i].item()
                best_iou, best_p = 0.0, -1
                for p_i in range(len(pred_boxes)):
                    if p_i not in matched_preds and ious[g_i, p_i] > best_iou:
                        best_iou = ious[g_i, p_i].item()
                        best_p = p_i

                if best_iou >= 0.5 and best_p >= 0:
                    p_lbl = pred_labels[best_p].item()
                    matched_preds.add(best_p)
                    confusion[g_lbl, p_lbl] += 1
                    if p_lbl == g_lbl:
                        tp_count[g_lbl] += 1
                    else:
                        fp_count[p_lbl] += 1
                else:
                    confusion[g_lbl, 0] += 1

            # Count unmatched predictions as false positives
            for p_i in range(len(pred_boxes)):
                if p_i not in matched_preds:
                    fp_count[pred_labels[p_i].item()] += 1

    print("\n" + "=" * 65)
    print("                MODEL EVALUATION RESULTS (IoU >= 0.5, Conf >= 0.5)")
    print("=" * 65)
    print(f"{'Class Name':18s} | {'GT Count':8s} | {'TP':5s} | {'FP':5s} | {'Recall':8s} | {'Precision':9s}")
    print("-" * 65)
    for i in range(1, 7):
        gt = total_gt[i]
        tp = tp_count[i]
        fp = fp_count[i]
        rec = (tp / gt * 100) if gt > 0 else 0
        prec = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0
        print(f"{class_names[i]:18s} | {gt:8d} | {tp:5d} | {fp:5d} | {rec:7.1f}% | {prec:8.1f}%")
    print("=" * 65)
    
    print("\n--- Key Diagnostics ---")
    print(f"Total images with predicted flowers: {flower_count_with_preds}")
    print(f"Images where flower was also predicted as a LEAF: {flower_leaf_collisions} ({flower_leaf_collisions/max(flower_count_with_preds,1)*100:.1f}%)")
    
    print("\n--- Confusion Breakdown (Ground Truth -> Predicted Class) ---")
    header = "GT \\ Pred        |" + "".join([f"{cn[:7]:>8s}" for cn in ['Missed'] + class_names[1:]])
    print(header)
    print("-" * len(header))
    for i in range(1, 7):
        row_str = f"{class_names[i]:16s} |"
        for j in range(7):
            row_str += f"{confusion[i, j]:8d}"
        print(row_str)

if __name__ == '__main__':
    main()

