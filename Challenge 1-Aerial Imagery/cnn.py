import os
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# ---------------------------------------------------------------------
# 1) READ CSV DATA
# ---------------------------------------------------------------------
train_df = pd.read_csv("archive/train.csv")

# The 'id' column contains image file names, e.g., "00000.jpg"
# The 'has_cactus' column contains 0 or 1 (label)

print("Number of training data:", len(train_df))
print(train_df.head())

# ---------------------------------------------------------------------
# 2) PREPARE TRAIN & VALIDATION SPLIT
# ---------------------------------------------------------------------
# Split the data so we can measure performance on the validation set
from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(
    train_df,
    test_size=0.2,
    random_state=42,
    stratify=train_df["has_cactus"]
)

print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}")

# ---------------------------------------------------------------------
# 3) IMAGE DATA GENERATOR
# ---------------------------------------------------------------------
# Use ImageDataGenerator for simple augmentation
# and also to simplify loading data from the folder
train_datagen = ImageDataGenerator(
    rescale=1./255,         # normalization [0..1]
    horizontal_flip=True,   # horizontal flip augmentation
    vertical_flip=False,    # can try True
    rotation_range=15,      # rotation
    width_shift_range=0.1,
    height_shift_range=0.1
)
val_datagen = ImageDataGenerator(rescale=1./255)

# path to train images folder
train_dir = "archive/train/train/"
# path to test images folder
test_dir = "archive/test/test/"

# flow_from_dataframe requires a dataframe with at least the columns: filename, class
# Here, the 'id' column -> 'filename', and 'has_cactus' -> label
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_dir,
    x_col='id',
    y_col='has_cactus',
    target_size=(32, 32),   # image size
    batch_size=32,
    class_mode='raw',       # since our labels are 0/1, choose 'raw'
    shuffle=True
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=train_dir,
    x_col='id',
    y_col='has_cactus',
    target_size=(32, 32),
    batch_size=32,
    class_mode='raw',
    shuffle=False
)

# ---------------------------------------------------------------------
# 4) BUILD A SIMPLE CNN MODEL
# ---------------------------------------------------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # output 0/1
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ---------------------------------------------------------------------
# 5) TRAINING
# ---------------------------------------------------------------------
# Train the model for several epochs, e.g., 10 or more
EPOCHS = 10

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator
)

# ---------------------------------------------------------------------
# 6) EVALUATION / VISUALIZATION OF RESULTS (OPTIONAL)
# ---------------------------------------------------------------------
import matplotlib.pyplot as plt

plt.figure()
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# ---------------------------------------------------------------------
# 7) PREDICT TEST DATA
# ---------------------------------------------------------------------
# For example, if the test set has no labels, we only need predictions.
# First, create a dataframe containing the list of file names in the test folder:
test_filenames = sorted(os.listdir(test_dir))
# if there are files other than jpg images, filter them here

test_df = pd.DataFrame({"id": test_filenames})

test_generator = val_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=test_dir,
    x_col='id',
    y_col=None,  # no labels
    target_size=(32, 32),
    batch_size=32,
    class_mode=None,
    shuffle=False  # do not shuffle
)

preds = model.predict(test_generator)
# Since we use sigmoid in the output, preds values are in the range [0..1]
# We can use a threshold of 0.5
pred_labels = (preds > 0.5).astype(int).reshape(-1)

# Check the first 10 predictions
print("Example predictions on test data:")
for i in range(10):
    print(test_df.iloc[i]['id'], "->", pred_labels[i])

# ---------------------------------------------------------------------
# 8) SAVE PREDICTIONS (OPTIONAL)
# ---------------------------------------------------------------------
# If you want to create a submission file (similar to Kaggle):
submission_df = pd.DataFrame({
    "id": test_df["id"],
    "has_cactus": pred_labels
})
submission_df.to_csv("my_submission.csv", index=False)
print("\nFile 'my_submission.csv' has been saved successfully.")