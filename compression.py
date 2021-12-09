# Imports
import tensorflow as tf
tf.random.set_seed(666)

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

import matplotlib.pyplot as plt

# Gather Flowers dataset
train_ds, validation_ds = tfds.load(
    "tf_flowers",
    split=["train[:85%]", "train[85%:]"],
    as_supervised=True
)

# Image utils
SIZE = (224, 224)

AUTO = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 64

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
# A bit of augmentation
@tf.function
def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.resize_with_crop_or_pad(image, SIZE[0] + 6, SIZE[1] + 6) 
    image = tf.image.random_crop(image, size=[SIZE[0], SIZE[1], 3])
    return image, label

# Prepare augmented loaders
training_aug_loader = (
    train_ds
    .shuffle(1024)
    .map(scale_resize_image, num_parallel_calls=AUTO)
    .cache()
    .map(augment, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)

# Teacher model utility
base_model = MobileNetV2(weights="imagenet", include_top=False,
        input_shape=(224, 224, 3))
base_model.trainable = True

def get_teacher_model():
    inputs = layers.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(5)(x)
    classifier = models.Model(inputs=inputs, outputs=x)
    
    return classifier
)

# Define loss function and optimizer
loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5) # Low since we are fine-tuning

# Train the teacher model
teacher_model = get_teacher_model()
teacher_model.compile(loss=loss_func, optimizer=optimizer, metrics=["accuracy"])
teacher_model.fit(training_loader,
                  validation_data=validation_loader,
                  epochs=20)
teacher_model.save('teacher.h5')

# Student model utility
def get_student_model(deeper=False):
    student_model = models.Sequential()
    student_model.add(layers.Conv2D(64, (3, 3), 
        input_shape=(224, 224, 3), 
        activation="relu",
        kernel_initializer="he_normal"))
    student_model.add(layers.MaxPooling2D((4, 4)))
    
    student_model.add(layers.Conv2D(128, (3, 3), 
        activation="relu",
        kernel_initializer="he_normal"))
    
    if deeper:
        student_model.add(tf.keras.layers.MaxPooling2D((4, 4)))
        student_model.add(tf.keras.layers.Conv2D(256, (3, 3), 
            activation="relu",
            kernel_initializer="he_normal"))
    
    student_model.add(layers.GlobalAveragePooling2D())
    student_model.add(layers.Dense(512, activation='relu'))
    student_model.add(layers.Dense(5))

    return student_model

# Average the loss across the batch size within an epoch
train_loss = tf.keras.metrics.Mean(name="train_loss")
valid_loss = tf.keras.metrics.Mean(name="test_loss")

# Specify the performance metric
train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name="train_acc")
valid_acc = tf.keras.metrics.SparseCategoricalAccuracy(name="valid_acc")

# Define the training loop

def get_kd_loss(student_logits, teacher_logits, 
                true_labels, temperature,
                alpha, beta):
    teacher_probs = tf.nn.softmax(teacher_logits / temperature)
    kd_loss = tf.keras.losses.categorical_crossentropy(
        teacher_probs, student_logits / temperature, 
        from_logits=True)
    
    ce_loss = tf.keras.losses.sparse_categorical_crossentropy(
        true_labels, student_logits, from_logits=True)
    
    total_loss = (alpha * kd_loss) + (beta * ce_loss)
    return total_loss / (alpha + beta)

class Student(tf.keras.Model):
    def __init__(self, trained_teacher, student, 
                 temperature=5., alpha=0.9, beta=0.1):
        super(Student, self).__init__()
        self.trained_teacher = trained_teacher
        self.student = student
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta

    def train_step(self, data):
        images, labels = data
        teacher_logits = self.trained_teacher(images)
        
        with tf.GradientTape() as tape:
            student_logits = self.student(images)
            loss = get_kd_loss(student_logits, teacher_logits,  
                               labels, self.temperature,
                               self.alpha, self.beta)
        gradients = tape.gradient(loss, self.student.trainable_variables)
        # As mentioned in Section 2 of https://arxiv.org/abs/1503.02531
        gradients = [gradient * (self.temperature ** 2) for gradient in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))

        train_loss.update_state(loss)
        train_acc.update_state(labels, tf.nn.softmax(student_logits))
        t_loss, t_acc = train_loss.result(), train_acc.result()
        train_loss.reset_states(), train_acc.reset_states()
        return {"train_loss": t_loss, "train_accuracy": t_acc}

    def test_step(self, data):
        images, labels = data
        teacher_logits = self.trained_teacher(images)
        
        student_logits = self.student(images, training=False)
        loss = get_kd_loss(student_logits, teacher_logits,  
                               labels, self.temperature,
                               self.alpha, self.beta)
        
        valid_loss.update_state(loss)
        valid_acc.update_state(labels, tf.nn.softmax(student_logits))
        v_loss, v_acc = valid_loss.result(), valid_acc.result()
        valid_loss.reset_states(), valid_acc.reset_states()
        return {"loss": v_loss, "accuracy": v_acc}

student = Student(teacher_model, get_student_model())
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
student.compile(optimizer)

student.fit(training_aug_loader, # Use augmentation here
            validation_data=validation_loader,
            epochs=10)

# Serialize
student.student.save("student_small.h5")

