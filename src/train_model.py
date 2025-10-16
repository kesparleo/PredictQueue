import tensorflow as tf
from tensorflow.keras import layers, models

def train_nn(X_train, y_train, X_val, y_val, input_dim, epochs=30, batch_size=128):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="linear")
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=epochs, batch_size=batch_size, verbose=2)
    model.save("models/nn_atendimento.h5")
    print("✅ Modelo treinado e guardado em models/nn_atendimento.h5")
    return model
