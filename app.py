from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from model import OnnxModel, ImagePreprocessor
import numpy as np
import uvicorn
import tempfile

app = FastAPI()

# Load model and preprocessor at startup
onnx_model = OnnxModel("model.onnx")
preprocessor = ImagePreprocessor()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an image file, preprocesses it, runs inference, and returns the predicted class index.
    """
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        input_array = preprocessor.preprocess(tmp_path)
        output = onnx_model.predict(input_array)
        pred_class = int(np.argmax(output))
        return JSONResponse({"predicted_class": pred_class})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500) 