import numpy as np

def recommend_servers(current_index, df, model, past=15, horizon=5, max_c=6, queue_threshold=5):
    """
    current_index: índice atual no dataframe df
    df: dataframe com dados de simulação
    model: modelo LSTM treinado
    """
    seq = []
    window = df.iloc[current_index - past:current_index]
    minutes_per_day = 480

    for _, row in window.iterrows():
        seq.append([
            row["arrivals"],
            row["queue"],
            row["in_service"],
            row["servers"],
            np.sin(2*np.pi*row["minute_of_day"]/minutes_per_day),
            np.cos(2*np.pi*row["minute_of_day"]/minutes_per_day)
        ])
    
    seq = np.array([seq], dtype=np.float32)  # (1, past, features)
    
    for c in range(1, max_c+1):
        seq[0, -1, 4] = c  # substitui "servers" pelo número de atendentes candidato
        pred = float(model.predict(seq)[0,0])
        if pred <= queue_threshold:
            return c, pred
    
    return max_c, float(model.predict(seq)[0,0])
