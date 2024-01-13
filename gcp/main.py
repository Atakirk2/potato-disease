from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np

BUCKET_NAME = "capstone-plant-disease"
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
model = None

def download_blob(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

def predict(request):
    global model
    if model is None:
        print("Model started downloading")
        download_blob(BUCKET_NAME, "models/potatoes.h5", "/tmp/potatoes.h5")
        print("Model ended downloading")
    print("Trying to load model")
    model = tf.keras.models.load_model("/tmp/potatoes.h5")
    print("Model loaded successfully")
    image = request.files["file"]
    image = np.array(Image.open(image).convert("RGB").resize((256,256)))
    image = image/255
    image_array = tf.expand_dims(image,0)
    predictions = model.predict(image_array)
    print(predictions)
    prediction = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = round(100*(np.max(predictions[0])), 2)
    
    return ({"class": prediction, "confidence": confidence})