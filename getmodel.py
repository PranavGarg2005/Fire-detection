import urllib.request

url = "https://huggingface.co/arnabdhar/YOLOv8-Fire-and-Smoke/resolve/main/model.pt"
print("Downloading fire detection model...")
urllib.request.urlretrieve(url, "fire_model.pt")
print("Done! fire_model.pt saved.")