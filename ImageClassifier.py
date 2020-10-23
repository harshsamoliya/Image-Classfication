import numpy as np
from bing_image_downloader import downloader

# i used to black car images and red car images for classification


# downloader.download("redcar",limit=20,adult_filter_off=True,timeout=60)
# downloader.download("blackcar",limit=20,adult_filter_off=True,timeout=60)


import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imread
import os

target = []
images = []  # in  matrix format
flat_data = []  # image in vector format

DataDir = 'G:\python projects\MiniCarProject\color'
Categories = ['redcar', 'blackcar']

# for loop for all categories
# PreProccing :  importand


for category in Categories:
    class_num = Categories.index(category)
    path = os.path.join(DataDir, category)
    for img in os.listdir(path):
        img_array = imread(os.path.join(path, img))
        img_resized = resize(img_array, (150, 150, 3))
        flat_data.append(img_resized.flatten())
        images.append(img_resized)
        target.append(class_num)

target = np.array(target)
images = np.array(images)
flat_data = np.array(flat_data)

import pandas as pd

# creating a DataFrame
df = pd.DataFrame(flat_data)
# put target in dataframe
df['Target'] = target

# for printing the dataframe
# print(df)

# x = df.iloc[:, 1:67499].values
# y = df.iloc[:, 67500].values

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(flat_data,target, random_state=0)
#
# print(x_train.shape)
# print(x_test.shape)
#

from sklearn.neighbors import KNeighborsClassifier

# from sklearn.model_selection import GridSearchCV
#
# import numpy as np
# z = np.arange(1,19)
#
# param = {'n_neighbors':z}
# model = KNeighborsClassifier()
# model_grid = GridSearchCV(model,param)
# model_grid.fit(flat_data,target)
# print(model_grid.best_params_)
# value of k_neighbour in my case is "7" Best

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = KNeighborsClassifier(n_neighbors=7, metric='euclidean')
model.fit(x_train, y_train)

y_pred = model.predict(x_test)


from sklearn.metrics import accuracy_score,confusion_matrix
# print(accuracy_score(y_pred,y_test))
# accuracy is about 87.5%


# confusion matrix
# print(confusion_matrix(y_pred,y_test))


# Testing a new Image of Red Car
flat_data = []
img = imread('G:\python projects\MiniCarProject\color\\redcar\images.jpg')
img_resized = resize(img,(150,150,3))
flat_data.append(img_resized.flatten())
flat_data = np.array(flat_data)

y_out = model.predict(flat_data)
output = Categories[y_out[0]]
print(output)

# Testing a new Image of Black Car
flat_data = []
img = imread('G:\python projects\MiniCarProject\color\\blackcar\Image_3.jpg')
img_resized = resize(img,(150,150,3))
flat_data.append(img_resized.flatten())
flat_data = np.array(flat_data)

y_out = model.predict(flat_data)
output = Categories[y_out[0]]
print(output)
