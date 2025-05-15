# custom_relu.py
import torch
import torchvision.models as models
from torch.nn import Module
import time
from PIL import Image
import requests
from io import BytesIO
import torchvision.transforms as transforms


def load_labels():
    # Load ImageNet class labels
    LABELS_URL = 'https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json'
    response = requests.get(LABELS_URL)
    labels = response.json()
    return labels

# Load and preprocess image
def load_image():
    # URL of a dog image (you can replace with any image URL)
    IMAGE_URL = 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg'
    
    response = requests.get(IMAGE_URL)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = preprocess(img)
    return input_tensor.unsqueeze(0).cuda()  # Add batch dimension


def infer():
    input = load_image()

    # Load original ResNet-50
    model = models.resnet50(pretrained=True).cuda().eval()

    # Warmup
    with torch.no_grad():
        _ = model(input)

    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        output = model(input)
    torch.cuda.synchronize()
    print(f"Inference time: {(time.time() - start) * 1000:.2f}ms")

    # Get predictions
    labels = load_labels()
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_ids = torch.topk(probabilities, 5)

    print("\nTop predictions:")
    for i in range(top5_prob.size(0)):
        print(f"{labels[top5_ids[i]]:>20s}: {top5_prob[i].item()*100:.2f}%")


infer()