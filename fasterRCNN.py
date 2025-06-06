import os
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import ToTensor
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from tqdm import tqdm

# ========== DATASET DEFINITION ==========
class ParkingDataset(torch.utils.data.Dataset):
    def _init_(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]

    def _len_(self):
        return len(self.images)

    def _getitem_(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.images[idx].replace('.jpg', '.txt').replace('.png', '.txt'))

        image = Image.open(image_path).convert("RGB")
        image_width, image_height = image.size

        boxes = []
        labels = []

        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            with open(label_path) as f:
                for line in f.readlines():
                    cls, x_center, y_center, w, h = map(float, line.strip().split())
                    xmin = (x_center - w / 2) * image_width
                    ymin = (y_center - h / 2) * image_height
                    xmax = (x_center + w / 2) * image_width
                    ymax = (y_center + h / 2) * image_height

                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(int(cls))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}

        if self.transform:
            image = self.transform(image)

        return image, target

# ========== COLLATE FUNCTION ==========
def collate_fn(batch):
    return tuple(zip(*batch))

# ========== EVALUATION FUNCTION ==========
def evaluate(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            print(f"Sample prediction labels: {outputs[0]['labels']}")

# ========== MAIN ==========
def main():
    transform = ToTensor()

    train_dataset = ParkingDataset("data/train/images", "data/train/labels", transform=transform)
    val_dataset = ParkingDataset("data/val/images", "data/val/labels", transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    model = fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 2  # background + parking spot (empty/occupied)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        total_loss = 0
        
        for images, targets in train_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {losses.item()}")

    torch.save(model.state_dict(), "parking_model.pth")
    evaluate(model, val_loader, device)
    visualize_prediction(model, val_dataset, device)

def visualize_prediction(model, dataset, device, num_images=5):
    model.eval()
    for i in range(num_images):
        image, _ = dataset[i]
        image_tensor = image.to(device).unsqueeze(0)
        with torch.no_grad():
            prediction = model(image_tensor)[0]

        # Convert tensor image to numpy
        img = TF.to_pil_image(image.cpu())
        plt.figure(figsize=(10, 6))
        plt.imshow(img)

        # Draw boxes
        boxes = prediction['boxes'].cpu()
        labels = prediction['labels'].cpu()
        for box, label in zip(boxes, labels):
            xmin, ymin, xmax, ymax = box
            color = "green" if label == 0 else "red"  # You decide what label 0/1 means
            plt.gca().add_patch(plt.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                edgecolor=color, facecolor='none', linewidth=2
            ))
            plt.text(xmin, ymin - 5, f"Label: {label}", color=color, fontsize=12, backgroundcolor='white')

        plt.title(f"Prediction {i + 1}")
        plt.axis('off')
        plt.savefig(f"prediction_{i}.png")
        plt.close()

if __name__ == "_main_":
    main()