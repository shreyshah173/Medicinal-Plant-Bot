from PIL import Image
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Flatten
from keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import pickle
import streamlit as st
import os
# import Disease 

st.title("Plant Detection")

st.write("This is a simple image classification web app to predict the plant disease")
 
# taking image input from user

labels_path1 = r"labels_leaf_new.pkl"
model_weights_path1 = r"diseased_prediction.h5"
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

picture = st.camera_input("Take a picture")

if picture:
    st.image(picture)

img = ""

# Fetch Model
def DiseaseOutput(image, labels_path, model_weights_path):
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
    # image = tf.io.read_file(image_path)

    # image = tf.image.decode_image(image, channels=3)
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


# printing image 
if file is None and picture is None:
    st.text("Please upload an image file")
elif file is None:
    file = picture
    # st.image(picture, use_column_width=True)
    st.write("taken input image")
    image = Image.open(file)
    img = image
    # st.image(image, use_column_width=True)
    st.write("")
    st.write("Classifying...")
    st.write("")
    # image = picture

    # st.image(img, use_column_width=True)


    labels_path = r"labels_plants.pkl"
    model_weights_path = r"new_plants1.h5"

    # Fetch Model
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
    model.add(Dense(40,activation='softmax'))

    model.load_weights(model_weights_path)


    # Transform Image
    new_shape = (128, 128)
    # image = tf.io.read_file(img)

    # image = tf.image.decode_image(image, channels=3)
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

    # image_folder = r"D:\MyProjects\SIH\beginagain\ImageToBase64-master\backup\backend\public\Images"
    # image_folder = r"D:\MyProjects\SIH\backup\backend\public\Images"


    # image_files = os.listdir(image_folder)
    # for image_name in image_files:
    #     image_path = os.path.join(image_folder, image_name)
    #     if os.path.isfile(image_path):
    #         os.remove(image_path)

    st.write(f"Predicted plant is :", ans)

else:
    image = Image.open(file)
    img = image
    st.image(image, use_column_width=True)
    st.write("")
    st.write("Classifying...")
    st.write("")
    # image = picture

    # st.image(img, use_column_width=True)


    labels_path = r"labels_plants.pkl"
    model_weights_path = r"new_plants1.h5"

    # Fetch Model
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
    model.add(Dense(40,activation='softmax'))

    model.load_weights(model_weights_path)


    # Transform Image
    new_shape = (128, 128)
    # image = tf.io.read_file(img)
    disease = DiseaseOutput(img, labels_path1, model_weights_path1)
    st.write(f"Predicted disease is :", disease)

    # image = tf.image.decode_image(image, channels=3)
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

    # image_folder = r"D:\MyProjects\SIH\beginagain\ImageToBase64-master\backup\backend\public\Images"
    # image_folder = r"D:\MyProjects\SIH\backup\backend\public\Images"


    # image_files = os.listdir(image_folder)
    # for image_name in image_files:
    #     image_path = os.path.join(image_folder, image_name)
    #     if os.path.isfile(image_path):
    #         os.remove(image_path)

    st.write(f"Predicted plant is :", ans)



