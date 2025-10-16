import numpy as np
import pandas as pd

def build_supervised_lstm(df, past=15, horizon=5, minutes_per_day=480):
    """
    Constrói sequências para LSTM.
    X: (amostras, timesteps, features)
    y: (amostras,)
    """
    sequences = []
    targets = []

    feature_cols = ["arrivals", "queue", "in_service", "servers"]

    for idx in range(past, len(df) - horizon):
        # Pegamos a sequência completa dos últimos `past` minutos
        window = df.iloc[idx - past:idx]

        # Para cada minuto, pegamos features brutas + hora do dia como sen/cos
        seq = []
        for _, row in window.iterrows():
            seq.append([
                row["arrivals"],
                row["queue"],
                row["in_service"],
                row["servers"],
                np.sin(2 * np.pi * row["minute_of_day"] / minutes_per_day),
                np.cos(2 * np.pi * row["minute_of_day"] / minutes_per_day)
            ])
        
        sequences.append(seq)
        targets.append(df.iloc[idx + horizon]["queue"])

    X = np.array(sequences, dtype=np.float32)   # (amostras, timesteps, features)
    y = np.array(targets, dtype=np.float32)     # (amostras,)
    return X, y
