import pandas as pd
import numpy as np
import joblib
import time
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 1. Load Data
print("Loading data...")
# Assuming these files are in the same directory
datax = pd.read_csv('train_features.csv')
datay = pd.read_csv('train_labels.csv')

# 2. Select Features (Matching your script: 0,2,3,4,5,6,15,16)
# dt, src, dst, pktcount, bytecount, dur, Protocol, port_no
feature_rnn_columns = [0, 2, 3, 4, 5, 6, 15, 16]
X = datax.iloc[:, feature_rnn_columns].values
y = datay.iloc[:, 0].values

# 3. Scale Data
print("Scaling data...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Reshape for LSTM (samples, time_steps, features)
# Reshape to (Batch_Size, 1, 8)
X_rnn = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))

# 5. Build LSTM Model
print("Building LSTM...")
model = Sequential()
# Input shape is (1, 8)
model.add(LSTM(64, return_sequences=True, input_shape=(1, len(feature_rnn_columns))))
model.add(Dropout(0.2))
model.add(LSTM(32, return_sequences=False)) # Return False to get a 2D output for SVM
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu')) # Feature layer
model.add(Dense(1, activation='sigmoid')) # Classification layer (for pre-training)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 6. Train LSTM
print("Training LSTM (This may take a moment)...")
model.fit(X_rnn, y, epochs=5, batch_size=32, verbose=1)

# 7. Extract Features for SVM
# We create a new model that outputs the Dense(16) layer's data
print("Extracting features for SVM...")
feature_extractor = Model(inputs=model.inputs, outputs=model.layers[-2].output)
X_features = feature_extractor.predict(X_rnn)

# 8. Train SVM
print("Training SVM...")
svm = SVC(kernel='rbf')
svm.fit(X_features, y)

# 9. Save Everything
print("Saving models...")
model.save('rnn_model.h5')           # Save LSTM
joblib.dump(svm, 'svm_model.pkl')    # Save SVM
joblib.dump(scaler, 'scaler.pkl')    # Save Scaler

print("Done! Files 'rnn_model.h5', 'svm_model.pkl', and 'scaler.pkl' are ready for the app.")
