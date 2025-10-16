import numpy as np
import tensorflow as tf

def predict_next_queue(model, features):
    return float(model.predict(np.array([features], dtype=np.float32))[0, 0])

def recommend_servers(current_row, model, minutes_per_day=480, max_c=6, queue_threshold=5):
    base = [
        current_row["arrivals"],
        0.0,
        current_row["queue"],
        current_row["in_service"],
        None,
        np.sin(2 * np.pi * current_row["minute_of_day"] / minutes_per_day),
        np.cos(2 * np.pi * current_row["minute_of_day"] / minutes_per_day)
    ]
    for c in range(1, max_c + 1):
        base[4] = c
        pred = predict_next_queue(model, base)
        if pred <= queue_threshold:
            return c, pred
    return max_c, pred
