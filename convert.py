import torch
import torch.onnx
import torchvision
import timm

model_name = "efficientnet_b0"

device = "cuda" if torch.cuda.is_available else "cpu"
model = timm.create_model(model_name, pretrained=False, num_classes=2)
dummy_input = torch.randn(1, 3, 224, 224)
model.eval()
torch.onnx.export(model, dummy_input, f="efficientnet.onnx",
                  input_names=["input"], output_names=["output"])
