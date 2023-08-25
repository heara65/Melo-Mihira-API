import numpy
from numpy import array
from math import sqrt
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from tensorflow import keras

Savedmodel = keras.models.load_model('OverallNewModel1.h5')
# Savedmodel = keras.models.load_model('OverallNewModel.h5')
# Savedmodel = keras.models.load_model('C:/Users/User/Desktop/New Folder/2021-069-IT18154054/API/OverallNewModel1.h5')
print("Model Loaded")

genderdict = {'m': 0, 'f': 1}

countrylist = ['United States', 'Germany', 'United Kingdom', 'Poland', 'Russian Federation', 'Brazil', 'Sweden',
               'Spain', 'Finland', 'Netherlands', 'Italy', 'France', 'Canada', 'Australia', 'Turkey', 'Norway',
               'Czech Republic', 'Ukraine', 'Japan', 'Belgium', 'Mexico', 'Argentina', 'Switzerland', 'Austria',
               'Romania', 'Portugal', 'Bulgaria', 'Chile', 'Denmark', 'Colombia', 'Greece', 'Hungary', 'Latvia',
               'Slovakia', 'Croatia', 'Serbia', 'Lithuania', 'Estonia', 'Ireland', 'New Zealand', 'Belarus', 'Israel',
               'India', 'Venezuela', 'Indonesia', 'Singapore', 'Slovenia', 'Korea, Republic of', 'China',
               'South Africa', 'Malaysia', 'Philippines', 'Peru', 'Thailand', 'Moldova', 'Costa Rica', 'Iceland',
               'Taiwan', 'Paraguay', 'Bosnia and Herzegovina', 'Antarctica', 'Puerto Rico', 'Georgia', 'Macedonia',
               'Uruguay', 'Honduras', 'Barbados', 'Kazakhstan', 'Andorra', 'Saudi Arabia',
               'United States Minor Outlying Islands', 'Djibouti', 'Cocos (Keeling) Islands', 'Tunisia', 'Egypt',
               'Bolivia', 'Panama', 'Brunei Darussalam', 'Iran, Islamic Republic of', 'Dominican Republic',
               'El Salvador', 'Haiti', 'Ecuador', 'Guatemala', 'Morocco', 'Pakistan', 'Burkina Faso', 'Azerbaijan',
               'Cambodia', 'Hong Kong', 'Viet Nam', 'United Arab Emirates', 'Jamaica', 'Faroe Islands', 'Somalia',
               'Guinea-Bissau', 'Micronesia, Federated States of', 'Tuvalu', "Cote D'Ivoire", 'Libyan Arab Jamahiriya',
               'Nicaragua', 'Kyrgyzstan', 'Malta', 'Bermuda', 'Luxembourg', 'Kuwait', 'Cyprus',
               'Heard Island and Mcdonald Islands', 'Christmas Island', 'Cuba', 'Niue', 'Aruba', 'Vanuatu', 'Dominica',
               'Holy See (Vatican City State)', 'Uzbekistan', 'Bhutan', 'Montenegro', 'Reunion', 'Fiji',
               'Netherlands Antilles', 'Lebanon', 'Liechtenstein']
countrydict = dict(zip(countrylist, [i for i in range(len(countrylist))]))

ethnicityList = ['white', 'black', 'asian', 'indian', 'others']
ethnicitydict = dict(zip(ethnicityList, [i for i in range(len(ethnicityList))]))


# print(countrydict)

# Define a mapper functions
def mapr1(key):
    """ Maps numbers to categories (gender)"""
    return genderdict[key]


def mapr2(key):
    """ Maps numbers to categories (country)"""
    return countrydict[key]


def mapr3(key):
    """ Maps numbers to categories (country)"""
    return ethnicitydict[key]


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
# from keras.preprocessing import image
from PIL import Image
from matplotlib import pyplot
# from pprint import pprint
import keras
from keras.preprocessing import image


def finalImageOutput(path):
    imagess = []
    IMGpath = path

    # img = image.load_img(IMGpath)
    img = Image.open(IMGpath)

    # Emotion = input_image_Emotion(IMGpath)

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


