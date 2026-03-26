import numpy as np
import matplotlib.pyplot as plt
from config import N, K, T, Q, CBAR, BETA

# Load matrix
M = np.load("data/fixed_subscription_matrix.npy")

# Initialize
aoi = np.ones((N, K))
lambda_val = np.zeros(N)
avg_aoi_history = []

for t in range(T):
    # Increase AoI
    aoi[M == 1] += 1

    scores = np.zeros(N)

    for n in range(N):
        subs = np.where(M[n] == 1)[0]
        if len(subs) > 0:
            avg_aoi = aoi[n, subs].mean()
            scores[n] = avg_aoi - lambda_val[n]

    # Pick top topics
    chosen = np.argsort(scores)[-CBAR:]

    # Transmission
    for n in chosen:
        if np.random.rand() < Q:
            subs = np.where(M[n] == 1)[0]
            aoi[n, subs] = 1

    # Update lambda
    for n in range(N):
        lambda_val[n] = BETA * lambda_val[n] + (1 - BETA) * scores[n]

    avg_aoi = aoi[M == 1].mean()
    avg_aoi_history.append(avg_aoi)

# Plot
plt.plot(avg_aoi_history)
plt.xlabel("Time")
plt.ylabel("Average AoI")
plt.title("AoI over Time (Price-Based)")
plt.grid()

plt.savefig("results/aoi_price_based.png")
plt.show()