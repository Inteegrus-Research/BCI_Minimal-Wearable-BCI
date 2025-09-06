# train.py  — Train & save your 1D‐CNN once
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, iirnotch
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# Config
DATA_DIR   = 'data'                    # folder with your .csv EEG files
MODEL_DIR  = 'models'
MODEL_NAME = 'project3_cnn_model.keras'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

CHANNELS   = 8
FS         = 256       # Hz
WINDOW_SEC = 2.0       # seconds
STEP_SEC   = 0.5       # seconds

WIN_LEN = int(WINDOW_SEC * FS)
STEP    = int(STEP_SEC   * FS)

# Filters
def butter_bp(low, high, fs, order=4):
    ny = 0.5*fs
    return butter(order, [low/ny, high/ny], btype='band')

def notch(fs, freq=50.0, q=30.0):
    return iirnotch(freq/(0.5*fs), q)

def preprocess(sig):
    b, a = butter_bp(1,45,FS)
    s    = lfilter(b, a, sig)
    bn, an = notch(FS)
    s    = lfilter(bn, an, s)
    return s - np.mean(s)

# Build file list
files = sorted(os.path.join(DATA_DIR,f)
               for f in os.listdir(DATA_DIR)
               if f.lower().endswith('.csv'))

# Load & window data
X, y = [], []
for path in files:
    label = 0 if 'low' in path.lower() else 1
    df    = pd.read_csv(path, index_col=0)
    data  = df.values.T  # shape (channels, time)
    for start in range(0, data.shape[1]-WIN_LEN+1, STEP):
        w = data[:, start:start+WIN_LEN]
        proc = np.stack([preprocess(w[ch]) for ch in range(CHANNELS)])
        X.append(proc.T)
        y.append(label)

X = np.array(X)
y = np.array(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Model definition
model = Sequential([
    Input(shape=(WIN_LEN, CHANNELS)),
    Conv1D(16, 32, strides=8, padding='same', activation='relu'),
    MaxPooling1D(4),
    Conv1D(32, 16, strides=4, padding='same', activation='relu'),
    MaxPooling1D(4),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])
model.compile(
    optimizer=Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=16,
    verbose=2
)

# Save
os.makedirs(MODEL_DIR, exist_ok=True)
model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
