import tensorflow as tf
tf.enable_eager_execution()
import csv
import os
import numpy as np
import random
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers

def load_and_preprocess_image(image_path, target_size):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, expand_animations=False)

    img = tf.image.resize(img, target_size)
    
    img = img / 255.0
    
    return img.numpy()

num_classes = 14

all_images = []
all_labels = []

readed_labels = []

with open("./dbAnimeNewSelected.csv", encoding="utf-8") as csvDB:
    anime_reader = csv.reader(csvDB)
    i = -1
    for row in anime_reader:
        if i == -1:
            i += 1
            continue
        i += 1
        curr_labels = []
        if len(row) >= 20:  
            for j in range(10, 24):
                if row[j] == '':
                    curr_labels = []
                    break
                curr_labels.append(int(row[j]))
        readed_labels.append(curr_labels)
        if i > 5000:
            break

print("Readed labels")

taken = set()

for i in range(1000):
    j = random.randint(100, 5000)
    while j in taken:
        j = random.randint(100, 5000) 
    taken.add(j)
    img_path = "./IMG/" + str(j) + ".jpeg"
    if os.path.isfile(img_path):
        img = load_and_preprocess_image(img_path, (512, 768))
        if np.shape(img) != (512, 768, 3):
            continue
        curr_labels = readed_labels[j]
        if len(curr_labels) < num_classes:
            continue
        all_images.append(img)
        all_labels.append(curr_labels)
    if i % 100 == 0:
        print(i)
        
all_images = np.array(all_images)     
all_labels = np.array(all_labels)     

delim = round(len(all_images) * 0.8)

train_images = all_images[:delim]
train_labels = all_labels[:delim]

val_images = all_images[delim:]
val_labels = all_labels[delim:]

base_model = ResNet50(weights='imagenet', include_top=False,
                      input_shape=(512, 768, 3))

base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.5),
    layers.BatchNormalization(),
    layers.Dense(num_classes, activation='sigmoid')
])

loss_fn = tf.keras.losses.BinaryCrossentropy()
model.compile(optimizer='adam', loss=loss_fn,
              metrics=[tf.keras.metrics.CategoricalAccuracy()])

print(model.summary())

class_weights = {0: 1.2,
                 1: 1.2,
                 2: 0.25,
                 3: 0.6,
                 4: 0.4,
                 5: 0.2,
                 6: 0.25,
                 7: 1.2,
                 8: 0.5,
                 9: 0.35,
                 10: 0.5,
                 11: 0.9,
                 12: 0.7,
                 13: 1.2}

with tf.device('/GPU:0'):
    model.fit(train_images, train_labels, epochs=12,
              class_weight=class_weights,
              validation_data=(val_images, val_labels))
model.save('saved_model/model_seq_12epoch_1000data_catAcc.h5')

print("Done")