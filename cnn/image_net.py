# file: mnist_cnn_resnet_input.py
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------------- utils ----------------
def set_seed(seed=42):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True  # good for fixed-size batches

# ImageNet mean/std (ResNet-50 convention)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# -------------- models --------------
class CNNbyGPT(nn.Module):
    """
    Adapted to ResNet-50 input convention: expects [B, 3, 224, 224].
    We keep your block order; we just:
      - change first conv to in_channels=3,
      - add AdaptiveAvgPool2d(7,7) before Flatten so the FC stays 128*7*7 -> 256.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            # input: [B, 3, 224, 224]
            nn.Conv2d(3, 32, 3, padding=1),          # -> [B, 32, 224, 224]
            nn.ReLU(inplace=True),                    # -> [B, 32, 224, 224]

            nn.Conv2d(32, 64, 3, padding=1),         # -> [B, 64, 224, 224]
            nn.ReLU(inplace=True),                    # -> [B, 64, 224, 224]
            nn.MaxPool2d(2),                          # -> [B, 64, 112, 112]

            nn.Conv2d(64, 128, 3, padding=1),        # -> [B, 128, 112, 112]
            nn.ReLU(inplace=True),                    # -> [B, 128, 112, 112]
            nn.MaxPool2d(2),                          # -> [B, 128, 56, 56]

            # Make FC size stable across input sizes (mimic ResNet head behavior):
            nn.AdaptiveAvgPool2d((7, 7)),            # -> [B, 128, 7, 7]
            nn.Flatten(),                             # -> [B, 128*7*7] = [B, 6272]

            nn.Linear(128*7*7, 256), nn.ReLU(inplace=True),  # -> [B, 256]
            nn.Linear(256, num_classes),                    # -> [B, num_classes] (logits)
        )

        # one-shot shape print (optional)
        x = torch.randn(2, 3, 224, 224)
        for layer in self.net:
            x_old = x
            x = layer(x)
            print(f"{layer.__class__.__name__:<20} {tuple(x_old.shape)} -> {tuple(x.shape)}")

    def forward(self, x): 
        return self.net(x)


class MLPBasedNN(nn.Module):
    """
    If you really want an MLP-ish baseline with ResNet-sized input,
    downsample first to keep params sane:
      [B, 3, 224, 224] --(AdaptiveAvgPool2d(28,28))--> [B, 3, 28, 28]
      Flatten -> Linear(3*28*28 -> hidden) -> SiLU -> Linear(hidden -> classes)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.hidden = 1024
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d((224, 224)),    # -> [B, 3, 224, 224]
            nn.Flatten(),                       # -> [B, 3*224*224]
            nn.Linear(3*224*224, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, num_classes),
        )

    def forward(self, x): 
        return self.net(x)


# -------- ResNet-style bottleneck block (as in ResNet-50) --------
class Bottleneck(nn.Module):
    expansion = 4  # output channels = planes * expansion

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        # 1x1 reduce
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        # 3x3 spatial
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        # 1x1 expand
        out_planes = planes * self.expansion
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_planes)

        # projection for the skip path if shape/stride changes
        self.downsample = None
        if stride != 1 or in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))      # [B, planes,    H,   W]
        out = self.relu(self.bn2(self.conv2(out)))    # [B, planes,  H/s, W/s]
        out = self.bn3(self.conv3(out))               # [B, planes*4, H/s, W/s]

        if self.downsample is not None:
            identity = self.downsample(identity)      # match shape

        out += identity                                # residual add
        out = self.relu(out)
        return out


# -------- AdvancedCNN (ResNet-50-ish layout) --------
class AdvancedCNN(nn.Module):
    """
    ResNet-50 style:
      stem:   7x7 s=2 -> BN -> ReLU -> 3x3 maxpool s=2
      stages: [3, 4, 6, 3] bottleneck blocks with channel widths [64, 128, 256, 512]
      head:   global avg pool -> linear
    Input expected: [B, 3, 224, 224]  (ResNet family)
    """
    def __init__(self, num_classes=10):
        super().__init__()

        # ----- Stem -----
        # [B,3,224,224] -> [B,64,112,112] -> [B,64,56,56]
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),  # [B,64,112,112]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)                   # [B,64,56,56]
        )

        # ----- Stages (ResNet-50: 3,4,6,3 bottlenecks) -----
        self.layer1 = self._make_layer( in_planes=64,   planes=64,  blocks=3, stride=1)  # -> [B,256,56,56]
        self.layer2 = self._make_layer( in_planes=256,  planes=128, blocks=4, stride=2)  # -> [B,512,28,28]
        self.layer3 = self._make_layer( in_planes=512,  planes=256, blocks=6, stride=2)  # -> [B,1024,14,14]
        self.layer4 = self._make_layer( in_planes=1024, planes=512, blocks=3, stride=2)  # -> [B,2048,7,7]

        # ----- Head -----
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # [B,2048,1,1]
        self.fc = nn.Linear(2048, num_classes)       # [B,num_classes]

        # init (good defaults)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def _make_layer(self, in_planes, planes, blocks, stride):
        layers = [Bottleneck(in_planes, planes, stride=stride)]
        out_planes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(out_planes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # [B,3,224,224]
        x = self.stem(x)             # [B,64,56,56]
        x = self.layer1(x)           # [B,256,56,56]
        x = self.layer2(x)           # [B,512,28,28]
        x = self.layer3(x)           # [B,1024,14,14]
        x = self.layer4(x)           # [B,2048,7,7]
        x = self.avgpool(x)          # [B,2048,1,1]
        x = torch.flatten(x, 1)      # [B,2048]
        x = self.fc(x)               # [B,num_classes]
        return x

# -------------- data --------------
def get_loaders(bs=128, num_workers=4):
    """
    Produce 3×224×224, ImageNet-normalized tensors (ResNet-50 style).
    MNIST is grayscale; we convert to 3-channel to match ResNet inputs.
    """
    tfm_train = transforms.Compose([
        transforms.Resize((224, 224)),            # Resize MNIST to 224
        transforms.Grayscale(num_output_channels=3), # 1->3 channels
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tfm_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    train = datasets.MNIST(root="./data", train=True,  download=True, transform=tfm_train)
    test  = datasets.MNIST(root="./data", train=False, download=True, transform=tfm_test)

    common = dict(batch_size=bs, pin_memory=True, num_workers=num_workers,
                  persistent_workers=(num_workers>0), prefetch_factor=2)
    return (DataLoader(train, shuffle=True, **common),
            DataLoader(test,  shuffle=False, **common))

# -------------- eval --------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()  # <-- important for BatchNorm/Dropout behavior in future
    correct, total, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type=='cuda')):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return loss_sum/total, correct/total

# -------------- train --------------
def train(model, epochs=3, lr=2e-3, bs=128):
    """
    Slightly smaller batch by default since 224×224 uses more memory than 28×28.
    """
    set_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader, test_loader = get_loaders(bs=bs)

    # Optional compiler (guarded for PyTorch/py version combos)
    try:
        model = torch.compile(model)  # PyTorch 2.x; silently skip if unsupported
    except Exception as e:
        print(f"[info] torch.compile disabled: {e}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=='cuda'))

    best_acc, patience, bad = 0.0, 2, 0
    for ep in range(1, epochs+1):
        model.train()  # <-- important for BatchNorm updates in future
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type=='cuda')):
                logits = model(x)
                loss = F.cross_entropy(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        val_loss, val_acc = evaluate(model, test_loader, device)
        print(f"epoch {ep}: val_loss={val_loss:.4f} accuracy={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc, bad = val_acc, 0
            torch.save({'model': model.state_dict()}, "mnist_resnet_input.pt")
        else:
            bad += 1
            if bad > patience: break

if __name__ == "__main__":
    print("--------------------- MLP based model -------------------------")
    fc_mode = MLPBasedNN()

    train(fc_mode, epochs=3, lr=2e-3, bs=128)

    print("--------------------- Simpliest CNN-based model -------------------------")
    # More pixels → more compute; 3–5 epochs is fine for a quick check.
    cnn_model = CNNbyGPT()
    train(cnn_model, epochs=3, lr=2e-3, bs=128)
