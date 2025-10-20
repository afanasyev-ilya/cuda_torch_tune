# file: mnist_cnn.py
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def set_seed(seed=42):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True  # good for fixed-size batches

class CNNbyGPT(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            # input is [B, 1, 28, 28]
            nn.Conv2d(1, 32, 3, padding=1), # -> [B, 32, 28, 28], since in channels = 1, out channles = 32
            nn.ReLU(inplace=True), # eltwise same shape
            nn.Conv2d(32, 64, 3, padding=1), # -> [B, 64, 28, 28], since in channels = 32, out channles = 64
            nn.ReLU(inplace=True), # eltwise same shape
            nn.MaxPool2d(2), # -> [B, 64, 14, 14], 
            nn.Conv2d(64, 128, 3, padding=1), # -> [B, 128, 14, 14]
            nn.ReLU(inplace=True), # eltwise same shape
            nn.MaxPool2d(2),  # -> [B, 128, 7, 7]
            nn.Flatten(), # -> [B, 128*7*7]
            nn.Linear(128*7*7, 256), nn.ReLU(inplace=True), # [B, 256]
            nn.Linear(256, num_classes), # [B, 1]
        )

        x = torch.randn(4, 1, 28, 28)
        for i, layer in enumerate(self.net):
            x_old = x
            x = layer(x)
            print(f"{layer.__class__.__name__} {x_old.shape} -> {x.shape}")

    def forward(self, x): return self.net(x)

class FullyConnectedCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.hidden = 1024
        self.hidden2 = 1024
        self.net = nn.Sequential(
            nn.Flatten(), # used to convert nchv -> n, chw
            nn.Linear(28*28, self.hidden),
            #nn.ReLU(), # relu is eltwise x -> max(0, x), e.g. errases negative values. used here to prevent collapsed linear.
            nn.SiLU(),
            # collapsed linear means y1 = w1*x+b1, y2 = (y1)*x+b2 = (w1*x + b1)*w1 + b2 = (w1*w1)*x + (b2 + w1*b1) 
            nn.Linear(self.hidden, num_classes),
        )

    def forward(self, x): return self.net(x)

def get_loaders(bs=256, num_workers=4):
    tfm = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test  = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)
    common = dict(batch_size=bs, pin_memory=True, num_workers=num_workers,
                  persistent_workers=(num_workers>0), prefetch_factor=2)
    return (DataLoader(train, shuffle=True, **common),
            DataLoader(test,  shuffle=False, **common))

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type=='cuda'):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return loss_sum/total, correct/total

def train(epochs=3, lr=2e-3, bs=256):
    set_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_loaders(bs=bs)
    model = CNNbyGPT().to(device)

    model = torch.compile(model)  # PyTorch 2.x; remove if unavailable
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=='cuda'))

    best_acc, patience, bad = 0.0, 2, 0
    for ep in range(1, epochs+1):
        model.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type=='cuda'):
                logits = model(x)
                loss = F.cross_entropy(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        val_loss, val_acc = evaluate(model, test_loader, device)
        print(f"epoch {ep}: val_loss={val_loss:.4f} accuracy={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc, bad = val_acc, 0
            torch.save({'model': model.state_dict()}, "mnist_cnn.pt")
        else:
            bad += 1
            if bad > patience: break
    return model

if __name__ == "__main__":
    model = train(epochs=5, lr=2e-3, bs=256)
