import torch
from pytorch_model import Classifier, BasicBlock

# Path to the PyTorch weights file 
WEIGHTS_PATH = "./pytorch_model_weights.pth"
ONNX_PATH = "model.onnx"

# Instantiate the model
model = Classifier(BasicBlock, [2, 2, 2, 2])
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=torch.device('cpu')))
model.eval()

# Create a dummy input matching the model's input size
# (batch_size=1, channels=3, height=224, width=224)
dummy_input = torch.randn(1, 3, 224, 224)

# Export the model to ONNX
# Dynamic axes allow for variable batch size
torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11,
)

print(f"Model has been successfully converted to ONNX and saved as {ONNX_PATH}") 
