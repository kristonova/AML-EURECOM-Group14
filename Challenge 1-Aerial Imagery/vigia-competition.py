# ────────────────────────────────────────────────────────────────────
# 0. Imports  (add these once, near the top of the notebook)
# ────────────────────────────────────────────────────────────────────
import pandas as pd, numpy as np, tensorflow as tf
from sklearn.model_selection import StratifiedKFold         # CV splitter
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path

# ────────────────────────────────────────────────────────────────────
# 1. Global parameters  🖉 tweak as needed
# ────────────────────────────────────────────────────────────────────
IMG_SIZE     = (32, 32)        # input resolution in your CNN
BATCH_SIZE   = 64
EPOCHS       = 15
N_SPLITS     = 5               # k-fold
SEED         = 42

TRAIN_DIR    = "archive/train/train/"   # 🖉 path with labelled jpg
TEST_DIR     = "archive/test/test/"     # 🖉 path with un-labelled jpg
TRAIN_CSV    = "archive/train/train.csv"

# ────────────────────────────────────────────────────────────────────
# 2. Load labelled dataframe  (must have columns: id, has_cactus)
# ────────────────────────────────────────────────────────────────────
train_df = pd.read_csv(TRAIN_CSV)

# ────────────────────────────────────────────────────────────────────
# 3. Helper: build a fresh model each fold
# ────────────────────────────────────────────────────────────────────
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu',
                               input_shape=IMG_SIZE+(3,)),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1,  activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# ────────────────────────────────────────────────────────────────────
# 4. Cross-validation loop
# ────────────────────────────────────────────────────────────────────
skf             = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                                  random_state=SEED)
histories       = []
val_scores      = []
test_pred_table = []           # test probabilities per fold

for fold, (tr_idx, val_idx) in enumerate(skf.split(train_df['id'],
                                                   train_df['has_cactus'])):
    print(f"\n──────────  Fold {fold+1}/{N_SPLITS}  ──────────")

    # Split the master dataframe
    tr_fold_df  = train_df.iloc[tr_idx].reset_index(drop=True)
    val_fold_df = train_df.iloc[val_idx].reset_index(drop=True)

    # ── Generators (train gets augmentation, val/test are raw) ──
    tr_datagen  = ImageDataGenerator(rescale=1/255.,
                                     horizontal_flip=True,
                                     rotation_range=15)
    val_datagen = ImageDataGenerator(rescale=1/255.)
    test_datagen= ImageDataGenerator(rescale=1/255.)

    tr_gen = tr_datagen.flow_from_dataframe(
        tr_fold_df, TRAIN_DIR,
        x_col='id', y_col='has_cactus',
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='binary', shuffle=True, seed=SEED)

    val_gen = val_datagen.flow_from_dataframe(
        val_fold_df, TRAIN_DIR,
        x_col='id', y_col='has_cactus',
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='binary', shuffle=False)

    # Test generator (no labels, deterministic order)
    test_gen = test_datagen.flow_from_directory(
        directory = Path(TEST_DIR).parent,
        classes   = [Path(TEST_DIR).name],   # loads all jpg under test/
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        shuffle=False, class_mode=None)

    # ── Build & train model ──
    model = build_model()
    es = tf.keras.callbacks.EarlyStopping(patience=3,
                                          restore_best_weights=True,
                                          monitor='val_loss')

    hist = model.fit(tr_gen,
                     validation_data=val_gen,
                     epochs=EPOCHS,
                     callbacks=[es],
                     verbose=1)

    histories.append(hist)
    val_acc = hist.history['val_accuracy'][-1]
    val_scores.append(val_acc)
    print(f"Fold-{fold+1} final val-accuracy: {val_acc:.4f}")

    # ── Infer on test set ──
    test_probs = model.predict(test_gen, verbose=0).ravel()
    test_pred_table.append(test_probs)

# ────────────────────────────────────────────────────────────────────
# 5. Aggregate metrics & write submission
# ────────────────────────────────────────────────────────────────────
print("\nCV accuracy per fold:", np.round(val_scores,4))
print(f"Mean ± SD accuracy  : {np.mean(val_scores):.4f} ± {np.std(val_scores):.4f}")

# Ensemble (mean) of probabilities from all folds
test_mean_prob = np.mean(np.stack(test_pred_table, axis=0), axis=0)
test_labels    = (test_mean_prob > 0.5).astype(int)

submission = pd.DataFrame({
    "id": [Path(p).name for p in sorted(Path(TEST_DIR).glob("*.jpg"))],
    "has_cactus": test_labels
})
submission.to_csv("submission_kfold.csv", index=False)
print("✔ submission_kfold.csv saved")