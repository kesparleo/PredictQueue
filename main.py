import os

# Suprimir mensagens informativas do TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = tudo, 1 = avisos, 2 = erros apenas, 3 = erros fatais
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # desativa avisos oneDNN


from src.simulate import simulate_mm_c
from src.prepare_data import build_supervised_lstm as build_supervised
from src.train_model import train_nn
from src.recommend import recommend_servers
import pandas as pd
import numpy as np

if __name__ == "__main__":
    # 1. Simulação
    df = simulate_mm_c(total_minutes=8*60*60)  # 60 dias, 8h/dia

    # ---------- Novo bloco: gerar relatório PDF ----------
    from reports.generate_report import create_pdf_report
    create_pdf_report(df, filename="reports/relatorio_atendimento.pdf")
    # ------------------------------------------------------

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
    row_index = 1000
    c_rec, pred_q = recommend_servers(row_index, df, model)
    print(f"\n🔍 Recomendação: {c_rec} atendentes → previsão de fila = {pred_q:.2f}")
