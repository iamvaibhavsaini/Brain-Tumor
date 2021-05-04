import keras
from PIL import Image, ImageOps
import numpy as np


def teachable_machine_classification(img, weights_file):
    # Load the model
    model = keras.models.load_model('model.h5')

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 128,128, 3), dtype=np.float32)
    image = img
    #image sizing
    size = (128,128)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 255)

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    print(np.argmax(prediction))

    return np.argmax(prediction) # return position of the highest probability
