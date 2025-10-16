from src.simulate import simulate_mm_c
from src.prepare_data import build_supervised
from src.train_model import train_nn
from src.recommend import recommend_servers
import pandas as pd
import numpy as np

if __name__ == "__main__":
    # 1. Simulação
    df = simulate_mm_c(total_minutes=8*60*60)  # 60 dias, 8h/dia
    # 2. Preparar dados
    X, y = build_supervised(df)
    n = len(X)
    i1, i2 = int(0.7*n), int(0.85*n)
    X_train, y_train = X[:i1], y[:i1]
    X_val, y_val = X[i1:i2], y[i1:i2]
    X_test, y_test = X[i2:], y[i2:]

    # 3. Treinar modelo
    model = train_nn(X_train, y_train, X_val, y_val, X.shape[1])

    # 4. Testar recomendação
    row = df.iloc[1000]
    c_rec, pred_q = recommend_servers(row, model)
    print(f"\n🔍 Recomendação: {c_rec} atendentes → previsão de fila = {pred_q:.2f}")
