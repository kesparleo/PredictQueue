import numpy as np
import pandas as pd

def build_supervised(df, past=15, horizon=5, minutes_per_day=480):
    data = []
    for idx in range(past, len(df) - horizon):
        window = df.iloc[idx - past:idx]
        features = [
            window["arrivals"].mean(),
            window["arrivals"].std(),
            window["queue"].iloc[-1],
            window["in_service"].iloc[-1],
            window["servers"].iloc[-1],
            np.sin(2 * np.pi * df.iloc[idx]["minute_of_day"] / minutes_per_day),
            np.cos(2 * np.pi * df.iloc[idx]["minute_of_day"] / minutes_per_day)
        ]
        target = df.iloc[idx + horizon]["queue"]
        data.append((features, target))
    X = np.array([d[0] for d in data], dtype=np.float32)
    y = np.array([d[1] for d in data], dtype=np.float32)
    return X, y
