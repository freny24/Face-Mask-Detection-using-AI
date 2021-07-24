import pandas as pd
import numpy as np
import cv2
import json
import os
import matplotlib.pyplot as plt
import random
import seaborn as sns
from keras.models import Sequential
from keras import optimizers
from keras import backend as K
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
directory =r"C:\Users\Aryan\PycharmProjects\FACE MASK\face-mask-dataset\Dataset\test"
image_directory = r"C:\Users\Aryan\PycharmProjects\FACE MASK\face-mask-dataset\Dataset\test"
df = pd.read_csv("C:\Users\Aryan\PycharmProjects\FACE MASK\face-mask-dataset\Dataset\train")
df_test = pd.read_csv("C:\Users\Aryan\PycharmProjects\FACE MASK\face-mask-dataset\Dataset\train")

cvNet = cv2.dnn.readNetFromCaffe('weights.caffemodel')
def getJSON(filePathandName):
    with open(filePathandName,'r') as f:
        return json.load(f)
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)])
    return cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))