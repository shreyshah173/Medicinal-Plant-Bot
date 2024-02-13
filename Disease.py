import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Flatten
from keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import pickle

# Plants
# image_path = r"A:\sih\DISEASED INDENTIFICATION\DATA\Jatropha, status - healthy\0006_0003.JPG"
image_path = r"D:\MyProjects\SIH\test_data\Brahmi\923.jpg"
labels_path = r"D:\MyProjects\SIH\Aashit\AASHIT_SIH_WORK\labels_leaf_new.pkl"
model_weights_path = r"D:\MyProjects\SIH\Aashit\AASHIT_SIH_WORK\diseased_prediction.h5"

# Fetch Model
def DiseaseOutput(image_path, labels_path, model_weights_path):
    conv_base = VGG16(
        weights='imagenet',
        include_top = False,
        input_shape=(128,128,3)
    )

    model = Sequential()
    model.add(conv_base)
    model.add(Flatten())
    model.add(Dense(256,activation='relu'))
    model.add(Dense(128,activation='relu'))
    model.add(Dense(20,activation='softmax'))

    model.load_weights(model_weights_path)


    # Transform Image
    new_shape = (128, 128)
    image = tf.io.read_file(image_path)

    image = tf.image.decode_image(image, channels=3)
    resized_image = tf.image.resize(image, new_shape)

    resized_image = resized_image / 255.0
    resized_image = tf.expand_dims(resized_image, axis=0)


    # Predict
    with open(labels_path, 'rb') as file:
        labels = pickle.load(file)
        
    predictions = model.predict(resized_image)
    max_index = np.argmax(predictions)

    def find_key(dictionary, value):
        for key, val in dictionary.items():
            if val == value:
                return key
        return None

    ans = find_key(labels, max_index)

    # print("before return")
    return ans

# print("started printing ")
print(DiseaseOutput(image_path, labels_path, model_weights_path))   
# print("ended printing ")

