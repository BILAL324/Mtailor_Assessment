import requests
import os
import sys


def predict(api_url, image_path, api_key=None):
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    with open(image_path, "rb") as f:
        files = {"file": f}
        response = requests.post(api_url, files=files, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python test_server.py <api_url> <image_path> <api_key>")
        sys.exit(1)

    api_url = sys.argv[1]
    image_path = sys.argv[2]
    api_key = sys.argv[3]

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        sys.exit(1)

    result = predict(api_url, image_path, api_key)
    print(f"Result: {result}")
