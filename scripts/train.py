import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from hurricane_intensity.dataset import TCDataset

TRAIN_STORMS = [
    "AL142018",  # Michael
    "AL142024",  # Milton
    "AL052019",  # Dorian
    "AL062020",  # Fay
    "AL092021",  # Ida
    "AL142021",  # Nicholas
    "AL102023",  # Idalia
    "AL012024",  # Alberto
    "AL042024",  # Debby
]

VAL_STORMS = [
    "AL192020",  # Sally
    "AL162023",  # Ophelia
    "AL132020",  # Laura
]

PROCESSED_ROOT = "data/processed"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

train_ds = TCDataset(PROCESSED_ROOT, TRAIN_STORMS)
val_ds = TCDataset(PROCESSED_ROOT, VAL_STORMS)
print(f"train samples: {len(train_ds)}  val samples: {len(val_ds)}")

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

model = resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(device)

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

NUM_EPOCHS = 20
best_val_loss = float("inf")

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        preds = model(images).squeeze(1)
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

    train_loss /= len(train_ds)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images).squeeze(1)
            val_loss += criterion(preds, labels).item() * images.size(0)

    val_loss /= len(val_ds)

    print(f"epoch {epoch+1}: train {train_loss*185:.2f} kt  val {val_loss*185:.2f} kt")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pt")
        print("  saved")