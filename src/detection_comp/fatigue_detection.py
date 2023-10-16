import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.layers import Dense, Input, Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten

DATASET_PATH: str = 'eyes_labeled.csv'
OPEN_LABEL: str = 'Open_Eyes'
CLOSED_LABEL: str = 'Closed_Eyes'

df = pd.read_csv(DATASET_PATH)

X = []
Y = []

for label, label_num in [
    (OPEN_LABEL, 1),
    (CLOSED_LABEL, 0)
]:
    label_images = [np.array(Image.open(i).resize((64, 64))) for i in df.loc[df['label'] == label, 'image_path']]
    X += label_images
    Y += [label_num] * len(label_images)

X = (np.array(X) - np.min(X)) / (np.max(X) - np.min(X))
X = X / 255.0
Y = (np.array(Y) - np.min(Y)) / (np.max(Y) - np.min(Y))
X = np.expand_dims(X, -1)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
model = tf.keras.models.Sequential([
    Input(shape=(64, 64, 1)),

    Conv2D(filters=32, kernel_size=5, strides=1, activation='relu'),
    Conv2D(filters=32, kernel_size=5, strides=1, activation='relu', use_bias=False),
    BatchNormalization(),
    MaxPooling2D(strides=2),
    Dropout(0.3),

    Conv2D(filters=64, kernel_size=3, strides=1, activation='relu'),
    Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', use_bias=False),
    BatchNormalization(),
    MaxPooling2D(strides=2),
    Dropout(0.3),

    Flatten(),
    Dense(units=256, activation='relu', use_bias=False),
    BatchNormalization(),

    Dense(units=128, use_bias=False, activation='relu'),

    Dense(units=84, use_bias=False, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(units=1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='bestModel.h5',
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=1)
model.fit(x_train, y_train, validation_split=0.2, epochs=15, batch_size=32, callbacks=callback)

model.evaluate(x_test, y_test)
best_model = load_model('bestModel.h5')
best_model.evaluate(x_test, y_test)

for i in x_test[0:5]:
    result = best_model.predict(np.expand_dims(i, 0))
    plt.imshow(i)
    plt.show()

    if result > 0.5:
        print(OPEN_LABEL)
    else:
        print(CLOSED_LABEL)
