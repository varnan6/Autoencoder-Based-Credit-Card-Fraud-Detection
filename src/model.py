# model.py
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

def build_autoencoder(input_dim, encoding_dim=14, activation='relu', optimizer='adam'):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation=activation)(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer=optimizer, loss='mse')
    return autoencoder