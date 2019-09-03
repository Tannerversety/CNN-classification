import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import seaborn as sns
import os, sys
from tqdm import tqdm
import pandas as pd

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize


from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers import Input, Dense
from keras.layers.core import Dropout, Activation
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.models import Model

import datetime
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import accuracy_score
from keras.preprocessing.image import ImageDataGenerator

from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input as preprocess_input_Xception

from keras.models import load_model

model_xception = Xception(weights='imagenet', include_top=False)


def input_branch(input_shape=None):
    size = int(input_shape[2] / 4)

    branch_input = Input(shape=input_shape)
    branch = GlobalAveragePooling2D()(branch_input)
    branch = Dense(size, use_bias=False, kernel_initializer='uniform')(branch)
    branch = BatchNormalization()(branch)
    branch = Activation("relu")(branch)
    return branch, branch_input


Xception_branch, Xception_input = input_branch(input_shape=(10, 10, 2048))

net = Dropout(0.34)(Xception_branch)
net = Dense(1024, use_bias=False, kernel_initializer='uniform')(net)
net = BatchNormalization()(net)
net = Activation("relu")(net)
net = Dropout(0.34)(net)
net = Dense(120, kernel_initializer='uniform', activation="softmax")(net)

model = Model(inputs=[Xception_input], outputs=[net])
model.summary()

model.load_weights('2019-03-18_dog_breed_model.h5')

name_f = np.load('names.npy')

X_validXception = np.zeros((1, 10, 10, 2048), dtype=np.float32)

img = load_img('chien.jpg', target_size=(299, 299))  # this is a PIL image

x_img = img_to_array(img)
x_img = x_img.reshape((1,) + x_img.shape)

imput_train_xception = preprocess_input_Xception(x_img.copy())
train_xception = model_xception.predict(imput_train_xception)  # ,batch_size=32)

X_validXception[0, :, :, :] = train_xception[0, :, :, :]

predictions = model.predict([X_validXception])

print ('voici la race du chien prédite par notre algorithme : {}'.format(name_f[int(np.argmax(predictions))]))