#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers, Sequential, models
import pathlib
from matplotlib.image import imread
from tensorflow.keras.utils import img_to_array,array_to_img
import random
import os
from os import listdir
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG19


# In[2]:


BATCH_SIZE = 32
IMAGE_SIZE = 256
CHANNELS=3
EPOCHS=10


# In[3]:


dataset= tf.keras.preprocessing.image_dataset_from_directory(
    "D:\potato disease\PlantVillage",
    seed=12,
    shuffle=True,
    image_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE
)


# In[4]:


class_names = dataset.class_names
class_names


# In[5]:


plt.figure(figsize=(20,20)) 
for image_batch, labels_batch in dataset.take(1):
    for i in range(12):
        ax = plt.subplot(3, 4, i + 1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        plt.title(class_names[labels_batch[i]])
        plt.axis("off")


# In[6]:


def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1

    ds_size = len(ds)

    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds


# In[7]:


train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)


# In[8]:


print("Size of Data is :{0} \nBatch size of Training Data is :{1}\nBatch size of Validation Data is :{2} \nBatch size of Testing Data is :{3} " .format(len(dataset), len(train_ds), len(val_ds), len(test_ds)))


# In[9]:


for image_batch,labels_batch in dataset.take(1):
  print(image_batch[0].numpy()/255)


# In[10]:


resize_and_rescale = tf.keras.Sequential([
  layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
  layers.Rescaling(1./255),
])


# In[11]:


data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
])


# In[12]:


train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)


# In[13]:


train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y)
).prefetch(buffer_size=tf.data.AUTOTUNE)


# In[14]:


input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 3

model = models.Sequential([
    resize_and_rescale,
    layers.Conv2D(32, kernel_size = (3,3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax'),
])

model.build(input_shape=input_shape)


# In[15]:


model.summary()


# In[16]:


model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)


# In[17]:


history = model.fit(
    train_ds,
    batch_size=BATCH_SIZE,
    validation_data=val_ds,
    verbose=1,
    epochs=EPOCHS,
)


# In[18]:


accuracy = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']
#graphs for accuracy and loss of training and validation data
plt.figure(figsize = (20,20))
plt.subplot(2,3,1)
plt.plot(range(EPOCHS), accuracy, label = 'Training Accuracy')
plt.plot(range(EPOCHS), val_acc, label = 'Validation Accuracy')
plt.legend(loc = 'lower right')
plt.title('Training and Validation Accuracy')


# In[19]:


plt.subplot(1,2,2)
plt.plot(range(EPOCHS), loss, label = 'Training Loss')
plt.plot(range(EPOCHS), val_loss, label = 'Validation Loss')
plt.legend(loc = 'upper right')
plt.legend(loc = 'upper right')
plt.legend(loc = 'upper right')
plt.legend(loc = 'upper right')
plt.title('Training and Validation Loss')


# In[21]:


print(classification_report(correct_labels,predicted_labels))


# In[22]:


base_model1 = VGG16(weights='imagenet', include_top=False, input_shape=(256,256,3))



# In[23]:


for layer1 in base_model1.layers:
    layer1.trainable = False



# In[24]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
model1 = Sequential()
model1.add(base_model1)
model1.add(Flatten())
model1.add(Dense(256, activation='relu'))
model1.add(Dropout(0.5))
model1.add(Dense(3, activation='softmax'))



# In[25]:


model1.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)



# In[26]:


model1.summary()


# In[27]:


history1 = model1.fit
  (
  train_ds,
  batch_size=BATCH_SIZE,
  validation_data=val_ds,
  verbose=1,
  epochs=EPOCHS,
)


# In[28]:


accuracy1 = history1.history['accuracy']
val_acc1 = history1.history['val_accuracy']

loss1 = history1.history['loss']
val_loss1 = history1.history['val_loss']
plt.figure(figsize = (20,20))
plt.subplot(2,3,1)
plt.plot(range(EPOCHS), accuracy1, label = 'Training Accuracy')
plt.plot(range(EPOCHS), val_acc1, label = 'Validation Accuracy')
plt.legend(loc = 'lower right')
plt.title('Training and Validation Accuracy')


# In[29]:


plt.subplot(1,2,2)
plt.plot(range(EPOCHS), loss1, label = 'Training Loss')
plt.plot(range(EPOCHS), val_loss1, label = 'Validation Loss')
plt.legend(loc = 'upper right')
plt.legend(loc = 'upper right')
plt.legend(loc = 'upper right')
plt.legend(loc = 'upper right')
plt.title('Training and Validation Loss')


# In[31]:


print(classification_report(correct_labels1,predicted_labels1))


# In[32]:


base_model2 = VGG19(weights='imagenet', include_top=False, input_shape=(256,256,3))


# In[33]:


for layer2 in base_model2.layers:
    layer2.trainable = False


# In[34]:


model2 = Sequential()
model2.add(base_model2)\]`+
model2.add(Flatten())
model2.add(Dense(256, activation='relu'))
model2.add(Dropout(0.5))
model2.add(Dense(3, activation='softmax'))


# In[35]:


model2.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)


# In[36]:


model2.summary()


# In[37]:


history2 = model2.fit(
  train_ds,
  batch_size=BATCH_SIZE,
  validation_data=val_ds,
  verbose=1,
  epochs=EPOCHS,)


# In[38]:


accuracy2 = history2.history['accuracy']
val_acc2 = history2.history['val_accuracy']

loss2 = history2.history['loss']
val_loss2 = history2.history['val_loss']
plt.figure(figsize = (20,20))
plt.subplot(2,3,1)
plt.plot(range(EPOCHS), accuracy2, label = 'Training Accuracy')
plt.plot(range(EPOCHS), val_acc2, label = 'Validation Accuracy')
plt.legend(loc = 'lower right')
plt.title('Training and Validation Accuracy')


# In[39]:


plt.subplot(1,2,2)
plt.plot(range(EPOCHS), loss2, label = 'Training Loss')
plt.plot(range(EPOCHS), val_loss2, label = 'Validation Loss')
plt.legend(loc = 'upper right')
plt.legend(loc = 'upper right')
plt.legend(loc = 'upper right')
plt.legend(loc = 'upper right')
plt.title('Training and Validation Loss')


# In[41]:


print(classification_report(correct_labels2,predicted_labels2))


# In[43]:


import numpy as np
for images_batch, labels_batch in test_ds.take(1):

  first_image = images_batch[0].numpy().astype('uint8')
  first_label = labels_batch[0].numpy()

  print("first image to predict")
  plt.imshow(first_image)
  print("actual label:", class_names[first_label])

  batch_prediction = model.predict(images_batch)
  print("predicted label:", class_names[np.argmax(batch_prediction[0])])
     


# In[ ]:




