import cv2
import numpy as np

def preprocess_image(file):
    image = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img
