import os
import cv2
import tensorflow as tf
from tensorflow.compat.v1.keras.models import load_model
import keras
import numpy as np
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
keras.backend.set_session(sess)

label = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C','D', 'E', 'F', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z']
for x in label:
    print("Dang test ", x)
    path_folder = r"Character_DataVal"
    path_file = os.path.join(path_folder, x)
    #print(path_file)
    list_file = os.listdir(os.path.expanduser(path_file))
    #print(list_file)

    model = load_model("my_model.h5")
    count = 0
    count_file = 0
    for i in list_file:
        count_file +=1
        #print(os.path.join(path_file, i))
        img = cv2.imread(os.path.join(path_file, i))
        #print(img.shape)
        box_img= cv2.resize(img, (38,38))
        #box_img_3=np.stack((box_img,)*3, -1)
        #print(box_img_3.shape)
        test= box_img.reshape(1,38,38,3)
        predict= model.predict(test)
        value= np.argmax(predict)
        #print(value, int(x))
        if label[value] == x:
            count = count + 1

    print("Predict: " + str(x) + ":" + str(count) + "/" + str(count_file))
