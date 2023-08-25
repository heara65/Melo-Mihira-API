import numpy
from numpy import array
from math import sqrt
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from tensorflow import keras

Savedmodel = keras.models.load_model('OverallNewModel1.h5')
print("Model Loaded")

genderdict = {'m': 0, 'f': 1}


ethnicityList = ['white', 'black', 'asian', 'indian', 'others']
ethnicitydict = dict(zip(ethnicityList, [i for i in range(len(ethnicityList))]))



# # Define a mapper functions
# def mapr1(key):
#     """ Maps numbers to categories (gender)"""
#     return genderdict[key]



# def mapr3(key):
#     """ Maps numbers to categories (country)"""
#     return ethnicitydict[key]


ID_GENDER_MAP = {0: 'male', 1: 'female'}
GENDER_ID_MAP = dict((g, i) for i, g in ID_GENDER_MAP.items())
ID_RACE_MAP = {0: 'white', 1: 'black', 2: 'asian', 3: 'indian', 4: 'others'}
RACE_ID_MAP = dict((r, i) for i, r in ID_RACE_MAP.items())


def GenderMapper(gender_):
    genders = ['m', 'f']
    if gender_ == 'female':
        return genders[1]
    else:
        return genders[0]


from skimage import io
from PIL import Image
from matplotlib import pyplot
import keras
from keras.preprocessing import image


def finalImageOutput(path):
    imagess = []
    IMGpath = path

 
    img = Image.open(IMGpath)

    img = img.resize((198, 198))
    img = np.array(img) / 255.0

    imagess.append(img)
    finalImage = np.array(imagess)

    custom = Savedmodel.predict(finalImage)

    max_age = 100
    age_pred = custom[0] * max_age
    Age = format(int(age_pred))

    gender_pred = custom[2]
    gender_pred = gender_pred.argmax(axis=-1)
    Gender = ID_GENDER_MAP[gender_pred[0]]

    race_pred = custom[1]
    race_pred = race_pred.argmax(axis=-1)
    Race = ID_RACE_MAP[race_pred[0]]


    return Age, Gender, Race


