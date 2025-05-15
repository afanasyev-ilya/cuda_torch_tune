# custom_relu.py
import torch
import torchvision.models as models
from torch.nn import Module
import time
from PIL import Image
import requests
from io import BytesIO
import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights


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
fused_bn_relu_ext = ext.load(
    name="fused_bn_relu_ext",  # Different name
    sources=["fused_bn_relu.cpp", "fused_bn_relu_kernel.cu"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=True
)

class CustomReLU(Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input):
        return custom_relu_ext.custom_relu_forward(input)

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

class FusedBNReLU(Module):
    def __init__(self, bn_layer):
        super().__init__()
        self.register_buffer('weight', bn_layer.weight.clone())
        self.register_buffer('bias', bn_layer.bias.clone())
        self.register_buffer('running_mean', bn_layer.running_mean.clone())
        self.register_buffer('running_var', bn_layer.running_var.clone())
        self.eps = bn_layer.eps
        
    def forward(self, x):
        return CUDA_EXT.fused_bn_relu_forward(
            x,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            self.eps
        )

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

def fuse_bn_relu(model):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Sequential):
            new_children = []
            skip_next = False
            for i, child in enumerate(module.children()):
                if skip_next:
                    skip_next = False
                    continue
                if isinstance(child, torch.nn.BatchNorm2d) and i+1 < len(module) and isinstance(module[i+1], torch.nn.ReLU):
                    new_children.append(FusedBNReLU(child))
                    skip_next = True
                else:
                    new_children.append(child)
            setattr(model, name, torch.nn.Sequential(*new_children))
        else:
            fuse_bn_relu(module)

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

def infer(batch_size = 32, custom_ops_list = [], benchmark_iters = 10):
    print("\n Using custom ops: ", custom_ops_list)
    input = load_image()

    # Load original ResNet-50
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).cuda().eval()

    # 3. Manually set to eval mode without fusion
    model = model.eval()

    verify_no_fusion = False
    if verify_no_fusion:
        # 4. Verify no Conv-BatchNorm fusion
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                print(f"Conv layer {name} remains separate")
            if isinstance(module, torch.nn.BatchNorm2d):
                print(f"BatchNorm layer {name} remains separate")

    if len(custom_ops_list) > 0:
        print("replacing torch ops with custom kernels!")

        if "fused_bn_relu" in custom_ops_list:
            print("fused bn and relu")
            fuse_bn_relu(model)
        else:
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

    print("Top predictions:")
    for i in range(top5_prob.size(0)):
        print(f"{labels[top5_ids[i]]:>20s}: {top5_prob[i].item()*100:.2f}%")

    input = torch.randn(batch_size, 3, 224, 224).cuda()

    # Benchmark
    avg_time = 0.0
    min_time = 0.0
    max_time = 0.0
    for iter in range(0, benchmark_iters):
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            output = model(input)
        torch.cuda.synchronize()
        cur_time = (time.time() - start) * 1000
        avg_time += cur_time / benchmark_iters
        if min_time == 0:
            min_time = cur_time
        else:
            min_time = min(cur_time, min_time)
        if max_time == 0:
            max_time = cur_time
        else:
            max_time = max(cur_time, max_time)

    #print(f"Inference min time: {min_time:.2f} ms")
    print(f"Inference avg time: {avg_time:.2f} ms")
    #print(f"Inference max time: {max_time:.2f} ms")
    print("\n\n")


bs = 256
iters = 10

infer(batch_size=bs, custom_ops_list=[], benchmark_iters=iters)
infer(batch_size=bs, custom_ops_list=["fused_bn_relu"], benchmark_iters=iters)
infer(batch_size=bs, custom_ops_list=["relu", "bn"], benchmark_iters=iters)

