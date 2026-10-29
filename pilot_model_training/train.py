import os
import glob
import torch
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from dataset import CanopyDataset
from model import get_instance_segmentation_model

class ToTensor:
    def __call__(self, image, target):
        return F.to_tensor(image), target

def collate_fn(batch):
    return tuple(zip(*batch))

def resolve_project_root():
    """Detects whether script is running inside code/ or project root."""
    if os.path.exists("dataset_processed"):
        return "."
    if os.path.exists(os.path.join("..", "dataset_processed")):
        return ".."
    return "."

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Executing training on: {device}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")

    root_dir = resolve_project_root()

    # Discover all XML annotations across all days
    xml_files = sorted(glob.glob(os.path.join(root_dir, "*.xml")))
    if not xml_files:
        print("Error: No XML files found in the directory.")
        return

    print(f"Discovered {len(xml_files)} annotation XML files:")
    for x in xml_files:
        print(f"  - {os.path.basename(x)}")

    # Initialize Dataset across all available days
    dataset = CanopyDataset(
        xml_source=xml_files,
        image_root_dir=root_dir,
        target_size=(1024, 768),
        transforms=ToTensor()
    )

    if len(dataset) == 0:
        print("Error: Dataset is empty. Check XML paths.")
        return

    # Reproducible 85% Train / 15% Validation split
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
    print(f"Dataset split: {len(train_set)} training samples | {len(val_set)} validation samples")

    train_loader = DataLoader(
        train_set,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_set,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    # 7 classes: 1 background + 6 canopy instances
    model = get_instance_segmentation_model(num_classes=7)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0025, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)

    save_path = os.path.join(root_dir, "canopy_maskrcnn.pth")
    best_loss = float('inf')
    epochs = 18

    print(f"\nStarting training for {epochs} epochs...")
    print("=" * 60)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_i, (images, targets) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            train_loss += losses.item()

        lr_scheduler.step()
        avg_train_loss = train_loss / max(len(train_loader), 1)

        # Validation Loss
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()

        avg_val_loss = val_loss / max(len(val_loader), 1)

        print(f"Epoch [{epoch + 1:02d}/{epochs:02d}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Checkpoint the best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  --> Best model updated and saved to {save_path} (Val Loss: {best_loss:.4f})")

    print("=" * 60)
    print(f"Training complete! Best model checkpoint saved to: {save_path}")

if __name__ == '__main__':
    main()
