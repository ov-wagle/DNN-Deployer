# Imports

import tensorflow as tf

tf.random.set_seed(666)


from tensorflow.keras.applications import MobileNetV2

from tensorflow.keras import models

from tensorflow.keras import layers


import tensorflow_datasets as tfds

tfds.disable_progress_bar()


import matplotlib.pyplot as plt


import numpy as np


# Gather Flowers dataset

train_ds, validation_ds = tfds.load(

    "tf_flowers",

    split=["train[:85%]", "train[85%:]"],

    as_supervised=True

)


# Image utils

SIZE = (224, 224)


AUTO = tf.data.experimental.AUTOTUNE

BATCH_SIZE = 8


@tf.function

def scale_resize_image(image, label):

    image = tf.image.convert_image_dtype(image, tf.float32)

    image = tf.image.resize(image, SIZE)

    return (image, label)


training_loader = (

    train_ds

    .shuffle(1024)

    .map(scale_resize_image, num_parallel_calls=AUTO)

    .cache()

    .batch(BATCH_SIZE)

    .prefetch(AUTO)

)


validation_loader = (

    validation_ds

    .map(scale_resize_image, num_parallel_calls=AUTO)

    .cache()

    .batch(BATCH_SIZE)

    .prefetch(AUTO)

)


# HeNormal = tf.keras.initializers.he_normal()


# Recreate the exact same model, including its weights and the optimizer

# new_model = tf.keras.models.load_model('student_bigger.h5', custom_objects={'HeNormal': HeNormal},compile=False)

new_model = tf.keras.models.load_model('teacher_model.h5', compile=False)


# Show the model architecture

new_model.summary()


print(np.argmax(new_model.predict(validation_loader), axis=1))

label_lst = []

for i, (image, label) in enumerate(validation_ds):

  label_lst.append(label.numpy())



print(label_lst)
