import tensorflow as tf
from tensorflow.keras import layers, models

def train_nn(X_train, y_train, X_val, y_val, epochs=30, batch_size=128):
    """
    Treina uma LSTM para previsão de fila.
    X_train: (amostras, timesteps, features)
    y_train: (amostras,)
    """
    timesteps = X_train.shape[1]
    features = X_train.shape[2]

    model = models.Sequential([
        layers.Input(shape=(timesteps, features)),
        layers.LSTM(64, return_sequences=False),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="linear")
    ])

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2
    )

    model.save("models/nn_atendimento.h5")
    print("✅ Modelo treinado e guardado em models/nn_atendimento.h5")

    return model
