import tensorflow as tf
import numpy as np
import os
from matplotlib import pyplot as plt

path_folder= os.getcwd()
print(path_folder)
path_train= os.path.join(path_folder, "Character_DataTrain")
path_val= os.path.join(path_folder, "Character_DataVal")


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding= 'same', input_shape= (38,38,3)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding= 'same'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
    
    tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding= 'same'),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding= 'same'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),

    tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding= 'same'),
    tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding= 'same'),
    tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding= 'same'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),

    tf.keras.layers.Conv2D(512, (3,3), activation='relu', padding= 'same'),
    tf.keras.layers.Conv2D(512, (3,3), activation='relu', padding= 'same'),
    tf.keras.layers.Conv2D(512, (3,3), activation='relu', padding= 'same'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation= 'relu'),
    tf.keras.layers.Dense(512, activation= 'relu'),
    tf.keras.layers.Dense(31, activation= 'softmax')]) 

model.summary()

train_datagen= tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)
train_generator = train_datagen.flow_from_directory(path_train,
                                                   target_size= (38, 38),
                                                   batch_size= 32,
                                                   class_mode= 'categorical')
val_datagen= tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)
val_generator = val_datagen.flow_from_directory(path_val,
                                                   target_size= (38, 38),
                                                   batch_size= 32,
                                                   class_mode= 'categorical')
"""labels = (train_generator.class_indices)
print(labels)
print(len(labels))"""

model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['acc'],
)

history= model.fit(
                train_generator,
                steps_per_epoch=len(train_generator),
                epochs=30,
                validation_data=val_generator,
                validation_steps=len(val_generator))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 30), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, 30), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 30), history.history["acc"], label="train_acc")
plt.plot(np.arange(0, 30), history.history["val_acc"], label="val_acc")
plt.title("Training Loss and Acc on VGG")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()

plt.show()
model.save("my_model.h5")