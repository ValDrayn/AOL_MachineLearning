import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import ToTensor
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import pandas as pd  # <-- Impor Pandas
import time  # <-- Impor Time
from tqdm import tqdm
from torchvision import transforms

from torchmetrics.detection import MeanAveragePrecision

is_available = torch.cuda.is_available()

if is_available:
    print(f"Jumlah GPU: {torch.cuda.device_count()}")
    print(f"Nama GPU: {torch.cuda.get_device_name(0)}")

class ParkingDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.images[idx].replace('.jpg', '.txt').replace('.png', '.txt'))
        image = Image.open(image_path).convert("RGB")
        image_width, image_height = image.size
        boxes = []
        labels = []
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            with open(label_path) as f:
                for line in f.readlines():
                    _, x_center, y_center, w, h = map(float, line.strip().split())
                    xmin = (x_center - w / 2) * image_width
                    ymin = (y_center - h / 2) * image_height
                    xmax = (x_center + w / 2) * image_width
                    ymax = (y_center + h / 2) * image_height
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(1)
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        if boxes.shape[0] == 0:
            target = {"boxes": boxes, "labels": labels}
        else:
            target = {"boxes": boxes, "labels": labels}
            
        if self.transform:
            image = self.transform(image)
            
        return image, target

def collate_fn(batch):
    return tuple(zip(*batch))

@torch.no_grad()
def calculate_metrics(model, dataloader, device):
    metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox', class_metrics=True)
    
    total_val_box_loss = 0
    total_val_cls_loss = 0
    
    model.train()
    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Hitung validation loss
        loss_dict = model(images, targets)
        total_val_box_loss += loss_dict['loss_box_reg'].item()
        total_val_cls_loss += loss_dict['loss_classifier'].item()
    
    model.eval() # Set ke mode eval untuk mendapatkan prediksi
    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        predictions = model(images)
        metric.update(predictions, targets)
        
    results = metric.compute()
    
    metrics_dict = {
        'metrics/precision(B)': results['map_50'].item(), # map_50 adalah proksi yang baik untuk presisi pada IoU 0.5
        'metrics/recall(B)': results['mar_100'].item(), # 'mar_100' (max recall for 100 detections) adalah proksi yang baik
        'metrics/mAP50(B)': results['map_50'].item(),
        'metrics/mAP50-95(B)': results['map'].item(),
        'val/box_loss': total_val_box_loss / len(dataloader),
        'val/cls_loss': total_val_cls_loss / len(dataloader),
    }

    return metrics_dict

def main():
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor()
    ])
    train_dataset = ParkingDataset("data/train/images", "data/train/labels", transform=transform)
    val_dataset = ParkingDataset("data/val/images", "data/val/labels", transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    model = fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 2 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    num_epochs = 10
    
    training_history = []
    
    print("Memulai Training...")
    for epoch in range(num_epochs):
        model.train()
        start_time = time.time()
        
        total_train_box_loss = 0
        total_train_cls_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for images, targets in progress_bar:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            total_train_box_loss += loss_dict['loss_box_reg'].item()
            total_train_cls_loss += loss_dict['loss_classifier'].item()

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            progress_bar.set_postfix(loss=f"{losses.item():.4f}")

        avg_train_box_loss = total_train_box_loss / len(train_loader)
        avg_train_cls_loss = total_train_cls_loss / len(train_loader)
        
        val_metrics = calculate_metrics(model, val_loader, device)
        
        epoch_time = time.time() - start_time
        
        lr_pg0 = optimizer.param_groups[0]['lr']
        lr_pg1 = optimizer.param_groups[1]['lr']
        lr_pg2 = optimizer.param_groups[2]['lr'] if len(optimizer.param_groups) > 2 else 0

        epoch_data = {
            'epoch': epoch + 1,
            'time': epoch_time,
            'train/box_loss': avg_train_box_loss,
            'train/cls_loss': avg_train_cls_loss,
            'train/dfl_loss': 0,  # Tidak ada di Faster R-CNN
            'metrics/precision(B)': val_metrics['metrics/precision(B)'],
            'metrics/recall(B)': val_metrics['metrics/recall(B)'],
            'metrics/mAP50(B)': val_metrics['metrics/mAP50(B)'],
            'metrics/mAP50-95(B)': val_metrics['metrics/mAP50-95(B)'],
            'val/box_loss': val_metrics['val/box_loss'],
            'val/cls_loss': val_metrics['val/cls_loss'],
            'val/dfl_loss': 0, # Tidak ada di Faster R-CNN
            'lr/pg0': lr_pg0,
            'lr/pg1': lr_pg1,
            'lr/pg2': lr_pg2,
        }
        
        training_history.append(epoch_data)
        
        # 💾 Simpan ke CSV di setiap epoch agar progres tidak hilang
        df = pd.DataFrame(training_history)
        df.to_csv("faster_rcnn_training_log.csv", index=False)
        
        print(f"\nEpoch {epoch+1} Selesai | mAP50: {epoch_data['metrics/mAP50(B)']:.4f} | Waktu: {epoch_time:.2f}s")

    print("Training Selesai.")
    print("Log disimpan di faster_rcnn_training_log.csv")

if __name__ == "__main__":
    main()