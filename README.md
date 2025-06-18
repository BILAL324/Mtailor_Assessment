# Image Classification Deployment Pipeline

## Overview
This project demonstrates how to deploy an image classification model (ResNet18-based, trained on ImageNet) as a serverless API. The pipeline includes converting a PyTorch model to ONNX format, serving it with a FastAPI application inside a Docker container, and deploying it on a GPU-enabled cloud platform (such as Cerebrium).

## Steps

1. **Clone the Repository**  
   Begin by cloning the project repository to your local machine.

2. **Install Dependencies**  
   Set up a Python virtual environment and install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Obtain Model Weights**  
   Download the pre-trained model weights and place them in the project root directory.

4. **Convert Model to ONNX**  
   Convert the PyTorch model to ONNX format for optimized inference:
   ```bash
   python convert_to_onnx.py
   ```

5. **Test Locally**  
   Run local tests to verify the ONNX model and preprocessing pipeline:
   ```bash
   python test.py
   ```

6. **Run FastAPI Server Locally**  
   Start the FastAPI server to serve predictions:
   ```bash
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```
   Test the API endpoint with a sample image:
   ```bash
   curl -X POST "http://localhost:8000/predict" -F "file=@sample_image.JPEG"
   ```

7. **Build and Run with Docker**  
   Build the Docker image and run the container:
   ```bash
   docker build -t image-classifier:latest .
   docker run -p 8000:8000 image-classifier:latest
   ```

8. **Deploy to Cloud Platform**  
   Deploy the Dockerized API to a GPU-enabled cloud platform. This can be done using a CLI tool or by pushing the Docker image to a registry and deploying via the platform's web interface.

   Example (using a CLI tool):
   ```bash
   cerebrium deploy --config-file cerebrium.toml
   ```

   Or, push to a registry:
   ```bash
   docker tag image-classifier:latest <dockerhub-username>/image-classifier:latest
   docker push <dockerhub-username>/image-classifier:latest
   ```

9. **Test Deployed Endpoint**  
   Use the provided script to test the deployed API:
   ```bash
   python3.10 test_server.py <api_url> <image_path> <api_key>
   ```

## Notes
- Ensure the Docker image is accessible to the deployment platform.
- The API expects a POST request to `/predict` with an image file.
- The response is a JSON object containing the predicted class index.
- For troubleshooting, check deployment logs and verify all dependencies are included.
