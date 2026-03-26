import numpy as np
import matplotlib.pyplot as plt
from config import N, K, T, Q, CBAR, BETA

# Load fixed matrix
M = np.load("data/fixed_subscription_matrix.npy")

# Initialize AoI
aoi = np.ones((N, K))
avg_aoi_history = []

for t in range(T):
    # Increase AoI
    aoi[M == 1] += 1

    # Round-robin scheduling
    start = t % N
    order = [(start + i) % N for i in range(N)]

    scores = np.zeros(N)

    for n in order:
        subs = np.where(M[n] == 1)[0]
        if len(subs) > 0:
            scores[n] = aoi[n, subs].mean()

    # Pick top CBAR topics
    chosen = np.argsort(scores)[-CBAR:]

    # Transmission
    for n in chosen:
        if np.random.rand() < Q:
            subs = np.where(M[n] == 1)[0]
            aoi[n, subs] = 1

    avg_aoi = aoi[M == 1].mean()
    avg_aoi_history.append(avg_aoi)

# Plot
plt.plot(avg_aoi_history)
plt.xlabel("Time")
plt.ylabel("Average AoI")
plt.title("AoI over Time (Round Robin)")
plt.grid()

plt.savefig("results/aoi_round_robin.png")
plt.show()