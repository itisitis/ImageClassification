import tensorflow as tf
import numpy as np
from datetime import datetime
import os

from tensorflow.keras.applications import MobileNet ,MobileNetV2 ,EfficientNetB4
from tensorflow.keras.layers import GlobalAveragePooling2D, Input , Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import relu, sigmoid, softmax
from tensorflow.keras.optimizers import Adam, RMSprop

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

tf.__version__

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=False,
                                                 verbose=1)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
#preprocessing and augmentation based on the model
train_datagen = ImageDataGenerator(
        horizontal_flip=True,
        validation_split=0.20)
train_generator = train_datagen.flow_from_directory(
        './images/',
        target_size=(380, 380),
        batch_size=16,
        class_mode='binary')

# train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         horizontal_flip=True)
# train_generator = train_datagen.flow_from_directory(
#         './img',
#         target_size=(128, 128),
#         batch_size=4,
#         class_mode='binary')

base_model = EfficientNetB4(
    input_shape=(380,380,3),
    include_top=False, weights='imagenet', input_tensor=None, pooling=None,
    classes=2, classifier_activation='sigmoid'
)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1,activation = sigmoid)(x)

model = Model(inputs=base_model.input, outputs=x)

model.summary()
#optimizers
adam = Adam(learning_rate=0.001)
rms = RMSprop(learning_rate=0.0001)

model.compile(loss='binary_crossentropy', optimizer=rms, metrics=['accuracy'])
batch_size = 16
len(train_generator.filenames)//batch_size
model.fit_generator(train_generator, 
          epochs=10,           
          shuffle=True,
          callbacks=[tensorboard_callback, cp_callback])
