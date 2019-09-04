# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 23:15:01 2019

@author: Vhasija
"""
import sys
import os
import keras
from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from time import time
sys.path.append(r'C:/Users/hasij/OneDrive/Desktop/graph_theory/final files')
import Red_particles
con_base = VGG16(weights='imagenet',include_top=False,input_shape=(150, 150, 3))
con_base.summary()


def prototyping():
    prototype = models.Sequential()
    prototype.add(con_base)
    prototype.add(layers.Flatten())
    prototype.add(layers.Dense(256, activation='relu'))
    prototype.add(layers.Dense(1, activation='sigmoid'))
    prototype.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
    prototype.summary()
    return prototype

def train_model():
    
    train_loc = 'C:/Users/hasij/OneDrive/Desktop/graph_theory/Project/Project/rustnonrust/train'
    validation_loc = 'C:/Users/hasij/OneDrive/Desktop/graph_theory/Project/Project/rustnonrust/validation'
    train_dataset = ImageDataGenerator(rescale=1./255,rotation_range=30,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')
    test_dataset = ImageDataGenerator(rescale=1./255)
    trainner = train_dataset.flow_from_directory(train_loc,target_size=(150, 150),batch_size=4,class_mode='binary')
    validator = test_dataset.flow_from_directory(validation_loc,target_size=(150, 150),batch_size=16,class_mode='binary')
    return trainner, validator

def main_module():
    model = prototyping()
    trainner, validator = train_model()
    tensor_board = keras.callbacks.TensorBoard(log_dir='output/{}'.format(time()))
    model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=2e-5),metrics=['acc'])
    model.fit_generator(trainner,steps_per_epoch=10,epochs=10,validation_data=validator,validation_steps=20,verbose=2,callbacks=[tensor_board])
    image_path = 'C:/Users/hasij/OneDrive/Desktop/graph_theory/Project/Project/rustnonrust/validation/rust/rust.75.jpg'
    input_image = image.load_img(image_path, target_size=(150, 150))
    image_test = image.img_to_array(input_image)
    image_test = image_test.reshape((1,) + image_test.shape)
    image_test =image_test.astype('float32') / 255
    rust_prob = model.predict(image_test)
    if (rust_prob > 0.50):
        print("This is a Rust image")
        depth = 15
        thresh_hold = 0.8
        distance = 5
        thresh = 0.07
        img = Red_particles.scale_image(image_path, maxsize=1000000)
        Red_particles.energy_gLCM(img,depth,thresh_hold,distance,thresh)
    else:
        print("This is a no Rust image")
    
    return
main_module()