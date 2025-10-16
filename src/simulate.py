import numpy as np
import pandas as pd
from tqdm import trange

def lambda_at_minute(minute_of_day, minutes_per_day, base=0.2):
    x = (minute_of_day / minutes_per_day) * 2*np.pi
    return base * (1 + 1.5 * np.sin(x))

def servers_at_minute(minute_of_day):
    if 180 <= minute_of_day <= 300:
        return 4
    else:
        return 3

def simulate_mm_c(total_minutes=8*60*30, minutes_per_day=8*60, mu=1/4.0, seed=42):
    np.random.seed(seed)
    queue = 0
    in_service = []
    records = []

    for t in trange(total_minutes, desc="Simulating"):
        minute_of_day = t % minutes_per_day
        lam = max(0.0, lambda_at_minute(minute_of_day, minutes_per_day))
        arrivals = np.random.poisson(lam)
        c = servers_at_minute(minute_of_day)
        queue += arrivals

        if len(in_service) < c:
            free = c - len(in_service)
            start = min(queue, free)
            for _ in range(start):
                st = max(1, int(np.random.exponential(1.0/mu) + 0.5))
                in_service.append(st)
                queue -= 1

        in_service = [s - 1 for s in in_service if s > 1]
        records.append({
            "t": t,
            "minute_of_day": minute_of_day,
            "arrivals": arrivals,
            "queue": queue,
            "in_service": len(in_service),
            "servers": c
        })

    df = pd.DataFrame(records)
    df.to_csv("data/sim_atendimento.csv", index=False)
    print("✅ Simulação concluída — dados guardados em data/sim_atendimento.csv")
    return df
