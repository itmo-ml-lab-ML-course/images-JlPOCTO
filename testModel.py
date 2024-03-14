import tensorflow as tf
import os
tf.enable_eager_execution()
import csv
from tensorflow.keras.preprocessing import image
import numpy as np

def load_and_preprocess_image(image_path, target_size):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, expand_animations=False)

    img = tf.image.resize(img, target_size)
    
    img = img / 255.0
    
    return img.numpy()

model = tf.keras.models.load_model("./saved_model/model_seq_20epoch_1500data_pres.h5",
                                   compile=False)
loss_fn = tf.keras.losses.BinaryCrossentropy()
model.compile(optimizer='adam', loss=loss_fn,
              metrics=[tf.keras.metrics.CategoricalCrossentropy()])

num_classes = 14
all_images = []
all_labels = []
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
        if len(curr_labels) > 0:
            img_path = "./IMG/" + str(j) + ".jpeg"
            if os.path.isfile(img_path):
                img = load_and_preprocess_image(img_path, (512, 768))
                if np.shape(img) != (512, 768, 3):
                    continue
                if len(curr_labels) < num_classes:
                    continue
                all_images.append(img)
                all_labels.append(curr_labels)
        if i > 100:
            break
        
all_images = np.array(all_images)     
all_labels = np.array(all_labels) 

results = model.evaluate(all_images, all_labels)
print('test loss, test acc:', results)
