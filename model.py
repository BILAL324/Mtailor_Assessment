import numpy as np
from PIL import Image
import onnxruntime as ort
from torchvision import transforms

class ImagePreprocessor:
    """
    Handles preprocessing of images for the ONNX model.
    """
    def __init__(self):
        self.resize = transforms.Resize((224, 224))
        self.crop = transforms.CenterCrop((224, 224))
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def preprocess(self, img_path: str) -> np.ndarray:
        """
        Loads an image from the given path and applies preprocessing steps.
        Returns a numpy array suitable for ONNX model input.
        """
        img = Image.open(img_path).convert('RGB')
        img = self.resize(img)
        img = self.crop(img)
        img = self.to_tensor(img)
        img = self.normalize(img)
        img = img.unsqueeze(0)  # Add batch dimension
        return img.numpy()

class OnnxModel:
    """
    Loads an ONNX model and performs inference.
    """
    def __init__(self, onnx_path: str):
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, input_array: np.ndarray) -> np.ndarray:
        """
        Runs inference on the input array and returns the output probabilities.
        """
        outputs = self.session.run([self.output_name], {self.input_name: input_array})
        return outputs[0] 