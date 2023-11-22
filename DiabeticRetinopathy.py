#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 12:18:31 2022

@author: echan
"""

import numpy as np
import tensorflow as tf
import ahnn as an


workspace = '/Volumes/ExtremeSSD/DeepLearning/Diabetic_Retinopathy'

ext = 'png'
data = an.ManageData('/Volumes/ExtremeSSD/DeepLearning/Diabetic_Retinopathy/Data/colored/colored_images', 
                      ext, 0.2, seed = 123)    
# data = an.ManageData('/Volumes/ExtremeSSD/DeepLearning/Diabetic_Retinopathy/tmp', 
#                      0.2, seed = 123)    



prjname = "TEST"

max_epochs = 400

def augment_using_ops(images, labels):
    images = tf.image.random_flip_left_right(images)
    images = tf.image.random_contrast(images, 0.7, 1.0)
    images = tf.image.random_brightness(images, 0.3)
    # images = images + tf.random.normal(shape=tf.shape(images),mean=0,stddev=2.0**0.5, dtype=tf.dtypes.int8) #Add Gauss noise
    return (images, labels)


train_ds = data.select_ds('training')

print("<< Training set samples >>")
for i in train_ds.take(3):
    print("HOHO = ",str(i.numpy(),'utf-8'))

print(data.classnames())

# data.breakdown(train_ds)

classweights = data.class_weights(train_ds)
print("class weights = ", classweights)

train_ds = data.balance_list(train_ds, workspace+'/tmpT')
# import sys
# sys.exit(1)
train_ds = data.dpmap(train_ds, 32, augment_using_ops)


val_ds   = data.select_ds('validation')
# val_ds = data.balance_list(val_ds, workspace+'/tmpV')
val_ds   = data.dpmap(val_ds, 32, augment_using_ops)

class_names = data.classnames()
print(class_names)


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
plt.show()
    
for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

normalization_layer = tf.keras.layers.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

AUTOTUNE = tf.data.AUTOTUNE


num_classes = len(class_names)

model_1 = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(30, 3, activation='tanh'),
    # tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(30, 3, activation='tanh'),
    # tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(20, 3, activation='tanh'),
    # tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(20, 3, activation='tanh'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(20, 3, activation='tanh'),
    tf.keras.layers.MaxPooling2D(),
    # tf.keras.layers.Conv2D(5, 5, activation='tanh'),
    # tf.keras.layers.MaxPooling2D(),
    # tf.keras.layers.Conv2D(5, 5, activation='tanh'),
    # tf.keras.layers.MaxPooling2D(),
    # tf.keras.layers.Conv2D(5, 5, activation='tanh'),
    # tf.keras.layers.MaxPooling2D(),
    # tf.keras.layers.Conv2D(3, 5, activation='tanh'),
    # tf.keras.layers.MaxPooling2D(),
    # tf.keras.layers.Conv2D(3, 5, activation='tanh'),
    # tf.keras.layers.MaxPooling2D(),
    # tf.keras.layers.Conv2D(3, 5, activation='tanh'),
    # tf.keras.layers.MaxPooling2D(),
    # tf.keras.layers.Conv2D(3, 5, activation='tanh'),
    # tf.keras.layers.MaxPooling2D(),
    # tf.keras.layers.Conv2D(3, 5, activation='tanh'),
    # tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(20, 3, activation='tanh'),
    tf.keras.layers.GlobalMaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(num_classes)
])

model_2 = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, 5, activation='tanh', name='C1'),
    tf.keras.layers.Conv2D(32, 5, activation='tanh', name='C2'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 5, activation='tanh', name='C3'),
    tf.keras.layers.GlobalMaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='tanh', name='D1'),
    tf.keras.layers.Dense(128, activation='tanh', name='D2'),
    tf.keras.layers.Dense(128, activation='tanh', name='D3'),
    tf.keras.layers.Dense(128, activation='tanh', name='D4'),
    tf.keras.layers.Dense(128, activation='tanh', name='D5'),
    tf.keras.layers.Dense(128, activation='tanh', name='D6'),
    tf.keras.layers.Dense(128, activation='tanh', name='D7'),
    tf.keras.layers.Dense(num_classes, name='OUT')
    ])


model = model_1



callbacklist = [
    # an.printLearningRate(),
    an.plotLearning(prjname, workspace),
    an.SaveModelWeights(prjname, workspace, 25)
    ]

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
    )



model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=max_epochs,
    callbacks = callbacklist,
    class_weight = classweights
    )


model.summary()



