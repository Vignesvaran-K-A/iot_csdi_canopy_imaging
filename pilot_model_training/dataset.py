import os
import glob
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image

# Robust mapping to handle all XML annotation spelling variations
LABEL_MAP = {
    "__background__": 0,
    "leaf": 1,
    "flower": 2,
    "bud": 3,
    "pre-mature-pod": 4,
    "pre_mature_pod": 4,
    "pre-mature pod": 4,
    "pre mature pod": 4,
    "pre_mature": 4,
    "pre-mature": 4,
    "pre mature": 4,
    "mature_pod": 5,
    "mature pod": 5,
    "post_mature_pod": 6,
    "post-mature-pod": 6,
    "post mature pod": 6
}

class CanopyDataset(Dataset):
    def __init__(self, xml_source, image_root_dir, target_size=(1024, 768), transforms=None):
        self.image_root_dir = image_root_dir
        self.target_size = target_size
        self.transforms = transforms
        self.images = []

        if isinstance(xml_source, list):
            xml_files = xml_source
        elif os.path.isdir(xml_source):
            xml_files = sorted(glob.glob(os.path.join(xml_source, "*.xml")))
        else:
            xml_files = [xml_source]

        print(f"Loading annotations from {len(xml_files)} file(s)...")

        for xml_file in xml_files:
            if not os.path.exists(xml_file):
                continue
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for img in root.findall('image'):
                # Include images even if empty (for background learning), but skip missing files
                self.images.append(img)

        print(f"Total images collected across datasets: {len(self.images)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_node = self.images[idx]
        raw_name = img_node.attrib['name'].replace('\\', '/')
        
        # Auto-remap folder paths if needed
        candidate1 = os.path.join(self.image_root_dir, raw_name)
        if os.path.exists(candidate1):
            img_path = candidate1
        else:
            alt_name = raw_name.replace("dataset_processed", "processed_images")
            candidate2 = os.path.join(self.image_root_dir, alt_name)
            if os.path.exists(candidate2):
                img_path = candidate2
            else:
                img_path = candidate1

        orig_w = int(img_node.attrib['width'])
        orig_h = int(img_node.attrib['height'])

        img = Image.open(img_path).convert("RGB")
        target_w, target_h = self.target_size
        img = img.resize((target_w, target_h), Image.BILINEAR)

        scale_x = target_w / orig_w
        scale_y = target_h / orig_h

        boxes = []
        labels = []
        masks = []

        for poly in img_node.findall('polygon'):
            label_name = poly.attrib['label'].strip().lower()
            if label_name not in LABEL_MAP:
                continue

            class_id = LABEL_MAP[label_name]
            pts_raw = poly.attrib['points'].strip().split(';')
            poly_pts = []
            for pt in pts_raw:
                if pt:
                    x, y = map(float, pt.split(','))
                    poly_pts.append([x * scale_x, y * scale_y])

            pts_array = np.array(poly_pts, dtype=np.int32)
            if len(pts_array) < 3:
                continue

            mask = np.zeros((target_h, target_w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts_array], 1)

            xmin, ymin = np.min(pts_array[:, 0]), np.min(pts_array[:, 1])
            xmax, ymax = np.max(pts_array[:, 0]), np.max(pts_array[:, 1])

            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(class_id)
                masks.append(mask)

        # Handle images with zero valid annotations (empty images / background)
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, target_h, target_w), dtype=torch.uint8)
            area = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64)
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
