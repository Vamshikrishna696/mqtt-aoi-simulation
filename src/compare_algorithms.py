import numpy as np
import matplotlib.pyplot as plt
from config import N, K, T, Q, CBAR, BETA

M = np.load("data/fixed_subscription_matrix.npy")

# ---------- ROUND ROBIN ----------
def run_round_robin():
    aoi = np.ones((N, K))
    history = []

    for t in range(T):
        aoi[M == 1] += 1
        start = t % N
        order = [(start + i) % N for i in range(N)]

        scores = np.zeros(N)
        for n in order:
            subs = np.where(M[n] == 1)[0]
            if len(subs) > 0:
                scores[n] = aoi[n, subs].mean()

        chosen = np.argsort(scores)[-CBAR:]

        for n in chosen:
            if np.random.rand() < Q:
                subs = np.where(M[n] == 1)[0]
                aoi[n, subs] = 1

        history.append(aoi[M == 1].mean())

    return history


# ---------- PRICE BASED ----------
def run_price_based():
    aoi = np.ones((N, K))
    lambda_val = np.zeros(N)
    history = []

    for t in range(T):
        aoi[M == 1] += 1

        scores = np.zeros(N)
        for n in range(N):
            subs = np.where(M[n] == 1)[0]
            if len(subs) > 0:
                avg = aoi[n, subs].mean()
                scores[n] = avg - lambda_val[n]

        chosen = np.argsort(scores)[-CBAR:]

        for n in chosen:
            if np.random.rand() < Q:
                subs = np.where(M[n] == 1)[0]
                aoi[n, subs] = 1

        for n in range(N):
            lambda_val[n] = BETA * lambda_val[n] + (1 - BETA) * scores[n]

        history.append(aoi[M == 1].mean())

    return history


# Run both
rr = run_round_robin()
pb = run_price_based()

# Plot
plt.plot(rr, label="Round Robin")
plt.plot(pb, label="Price-Based")

plt.xlabel("Time")
plt.ylabel("Average AoI")
plt.title("Comparison of Scheduling Algorithms")
plt.legend()
plt.grid()

plt.savefig("results/comparison.png")
plt.show()