# custom_relu.py
import torch
import torchvision.models as models
from torch.nn import Module
import time
from PIL import Image
import requests
from io import BytesIO
import torchvision.transforms as transforms

#####################################################################################################

# Custom CUDA extension module
import torch.utils.cpp_extension as ext

# Load custom ReLU extension
custom_relu_ext = ext.load(
    name="custom_relu_ext",  # Must be unique
    sources=["custom_relu.cpp", "custom_relu_kernel.cu"],
    extra_cuda_cflags=["-O3"],
    verbose=True
)

# Load custom BatchNorm extension
custom_batchnorm_ext = ext.load(
    name="custom_batchnorm_ext",  # Unique name
    sources=["custom_bn.cpp", "custom_bn_kernel.cu"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=True
)


# Load fused BN-ReLU extension
#fused_bn_relu_ext = ext.load(
#    name="fused_bn_relu_ext",  # Different name
#    sources=["fused_bn_relu.cpp", "fused_bn_relu_kernel.cu"],
#    extra_cuda_cflags=["-O3", "--use_fast_math"],
#    verbose=True
#)

class CustomBatchNorm2d(Module):
    def __init__(self, bn_layer):
        super().__init__()
        self.register_buffer('weight', bn_layer.weight.clone())
        self.register_buffer('bias', bn_layer.bias.clone())
        self.register_buffer('running_mean', bn_layer.running_mean.clone())
        self.register_buffer('running_var', bn_layer.running_var.clone())
        self.eps = bn_layer.eps
        
    def forward(self, x):
        return custom_batchnorm_ext.custom_batchnorm_forward(
            x,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            self.eps
        )

class CustomReLU(Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input):
        return custom_relu_ext.custom_relu_forward(input)

def replace_batchnorm(model):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.BatchNorm2d):
            setattr(model, name, CustomBatchNorm2d(module))
        else:
            replace_batchnorm(module)

def replace_relu(module):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.ReLU):
            setattr(module, name, CustomReLU())
        else:
            replace_relu(child)

#####################################################################################################

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

#####################################################################################################

def infer(batch_size = 32, custom_ops_list = []):
    input = load_image()

    # Load original ResNet-50
    model = models.resnet50(pretrained=True).cuda().eval()

    if len(custom_ops_list) > 0:
        print("replacing torch ops with custom kernels!")

        if "relu" in custom_ops_list:
            replace_relu(model)
            print("replaced RELU!")

        if "bn" in custom_ops_list:
            replace_batchnorm(model)
            print("replaced Batchnorm!")

    # Warmup and verify
    with torch.no_grad():
        output = model(input)

    # Get predictions
    labels = load_labels()
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_ids = torch.topk(probabilities, 5)

    print("\nTop predictions:")
    for i in range(top5_prob.size(0)):
        print(f"{labels[top5_ids[i]]:>20s}: {top5_prob[i].item()*100:.2f}%")

    input = torch.randn(batch_size, 3, 224, 224).cuda()

    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        output = model(input)
    torch.cuda.synchronize()
    print("Using custom ops: ", custom_ops_list)
    print(f"Inference time: {(time.time() - start) * 1000:.2f}ms")


infer(batch_size = 32, custom_ops_list = ["relu", "bn"])
infer(batch_size = 32, custom_ops_list = [])