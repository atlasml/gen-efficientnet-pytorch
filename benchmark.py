from gen_efficientnet import efficientnet_b0
from sotabench.image_classification import ImageNet
import torch
import torchvision.transforms as transforms
import PIL

model = torch.hub.load('rwightman/gen-efficientnet-pytorch', 
                       'efficientnet_b0', 
                       pretrained=True)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
input_transform = transforms.Compose([
    transforms.Resize(256, PIL.Image.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

ImageNet.benchmark(
    model=model,
    paper_model_name='EfficientNet-B0',
    paper_arxiv_id='1905.11946',
    paper_pwc_id='efficientnet-rethinking-model-scaling-for',
    input_transform=input_transform,
    batch_size=256,
    num_gpu=1
)
