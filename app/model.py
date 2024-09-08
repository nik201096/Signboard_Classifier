import numpy as np
import tensorflow as tf
from keras.models import load_model as keras_load_model
from PIL import Image
import io

def load_model():
    return keras_load_model('saved_models/keras_cifar10_trained_model.h5')

def preprocess_image(image, target_size):
    # Convert the image to grayscale and normalize
    image = Image.open(image)
    image = image.convert("L")  # Convert to grayscale
    image = image.resize(target_size)
    image = np.array(image)
    image = image / 128.0 - 1
    image = np.expand_dims(image, axis=(0, -1))
    return image

def predict_traffic_sign(model, image):

    d={ 0: 'Speed limit 20km/h',
        1: 'Speed limit 30km/h',
        2: 'Speed limit 50km/h',
        3: 'Speed limit 60km/h',
        4: 'Speed limit 70km/h',
        5: 'Speed limit 80km/h',
        6: 'End of speed limit 80km/h',
        7: 'Speed limit 100km/h',
        8: 'Speed limit 120km/h',
        9: 'No passing',
        10: 'No passing for vehicles over 3.5 metric tons',
        11: 'Rightofway at the next intersection',
        12: 'Priority road',
        13: 'Yield',
        14: 'Stop',
        15: 'No vehicles',
        16: 'Vehicles over 3.5 metric tons prohibited',
        17: 'No entry',
        18: 'General caution',
        19: 'Dangerous curve to the left',
        20: 'Dangerous curve to the right',
        21: 'Double curve',
        22: 'Bumpy road',
        23: 'Slippery road',
        24: 'Road narrows on the right',
        25: 'Road work',
        26: 'Traffic signals',
        27: 'Pedestrians',
        28: 'Children crossing',
        29: 'Bicycles crossing',
        30: 'Beware of ice/snow',
        31: 'Wild animals crossing',
        32: 'End of all speed and passing limits',
        33: 'Turn right ahead',
        34: 'Turn left ahead',
        35: 'Ahead only',
        36: 'Go straight or right',
        37: 'Go straight or left',
        38: 'Keep right',
        39: 'Keep left',
        40: 'Roundabout mandatory',
        41: 'End of no passing',
        42: 'End of no passing by vehicles over 3.5 metric tons'}


    processed_image = preprocess_image(image, target_size=(32, 32))
    prediction = model.predict(processed_image)
    class_idx = np.argmax(prediction, axis=1)[0]



    return d[class_idx]
