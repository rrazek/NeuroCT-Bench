# NeuroCT-Bench — training script
# Essential, reproducible implementation

import os, time, random, argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import pandas as pd
from collections import Counter

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='dataset')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--weights', type=str, default='imagenet', choices=['imagenet','none'])
parser.add_argument('--out_prefix', type=str, default='__PREFIX__')
args = parser.parse_args()

DATA_DIR = args.data_dir
BATCH_SIZE = args.batch_size
IMG_SIZE = (args.img_size, args.img_size)
EPOCHS = args.epochs
WEIGHTS = None if args.weights == 'none' else 'imagenet'
OUT_PREFIX = args.out_prefix

train_ds = image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset='training', seed=SEED,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, color_mode='grayscale'
)
val_ds = image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset='validation', seed=SEED,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, color_mode='grayscale'
)

def to_rgb(image, label):
    return tf.image.grayscale_to_rgb(image), label

train_ds = train_ds.map(to_rgb, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(to_rgb, num_parallel_calls=tf.data.AUTOTUNE)

data_augmentation = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.06),
], name='data_augmentation')

normalization = layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x,y: (normalization(x), y), num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(lambda x,y: (normalization(x), y), num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.map(lambda x,y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

counts = Counter()
for _, labels in train_ds.unbatch():
    counts[int(labels.numpy())] += 1
total = sum(counts.values())
class_weight = {i: total/(len(counts)*v) for i,v in counts.items()}

from tensorflow.keras.applications import EfficientNetB0
base_model = EfficientNetB0(weights=WEIGHTS, include_top=False, input_shape=IMG_SIZE + (3,))
base_model.trainable = False
inputs = keras.Input(shape=IMG_SIZE + (3,))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer=keras.optimizers.Adam(1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(OUT_PREFIX + '_best.h5', monitor='val_loss', save_best_only=True)
]

start_time = time.time()
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, class_weight=class_weight)
end_time = time.time()

# Evaluation
y_true = []
y_prob = []
for images, labels in val_ds:
    preds = model.predict(images, verbose=0).flatten()
    y_true.extend(labels.numpy())
    y_prob.extend(preds)

y_true = np.array(y_true)
y_prob = np.array(y_prob)
y_pred = (y_prob > 0.5).astype(int)

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true))>1 else float('nan')
cm = confusion_matrix(y_true, y_pred, labels=[0,1]) if len(np.unique(y_true))>1 else np.array([])
specificity = (cm.ravel()[0]/(cm.ravel()[0]+cm.ravel()[1])) if cm.size==4 else float('nan')

metrics = {
    'accuracy': acc, 'precision': prec, 'recall': rec, 'specificity': specificity,
    'f1': f1, 'auc': auc, 'training_time_s': end_time-start_time
}

print('\n--- Evaluation Results ---')
for k,v in metrics.items():
    print(f'{k:12} : {v:.4f}' if isinstance(v,float) else f'{k:12} : {v}')

print('\nClassification Report:\n', classification_report(y_true, y_pred, target_names=['Normal','Stroke'], zero_division=0))

pd.DataFrame({'y_true': y_true, 'y_prob': y_prob, 'y_pred': y_pred}).to_csv(OUT_PREFIX + '_val_predictions.csv', index=False)
pd.DataFrame([metrics]).to_csv(OUT_PREFIX + '_metrics.csv', index=False)
