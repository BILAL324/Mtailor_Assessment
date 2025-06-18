from model import OnnxModel, ImagePreprocessor
import numpy as np
import os

# Paths
ONNX_PATH = "model.onnx"
TEST_IMAGES = [
    ("n01440764_tench.jpeg", 0),
    ("n01667114_mud_turtle.JPEG", 35),
]

def test_model():
    preprocessor = ImagePreprocessor()
    model = OnnxModel(ONNX_PATH)
    all_passed = True
    for img_path, expected_class in TEST_IMAGES:
        if not os.path.exists(img_path):
            print(f"Test image not found: {img_path}")
            all_passed = False
            continue
        input_array = preprocessor.preprocess(img_path)
        output = model.predict(input_array)
        pred_class = int(np.argmax(output))
        print(f"Image: {img_path} | Predicted: {pred_class} | Expected: {expected_class}")
        if pred_class == expected_class:
            print("Test PASSED")
        else:
            print("Test FAILED")
            all_passed = False
    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests failed.")

if __name__ == "__main__":
    test_model() 