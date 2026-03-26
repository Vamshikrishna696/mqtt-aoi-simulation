import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

N = 7
K = 5
T = 300
q = 0.9

Cbar = 3
beta = 0.94
lam = 0.0

sub_prob = 0.5
M = (np.random.rand(N, K) < sub_prob).astype(int)

for n in range(N):
    if M[n].sum() == 0:
        M[n, np.random.randint(0, K)] = 1

aoi = np.zeros((N, K), dtype=float)
aoi[M == 1] = 1.0

lam_hist = np.zeros(T)
avg_aoi_hist = np.zeros(T)
tx_hist = np.zeros(T)

def topic_mean_aoi(n):
    subs = np.where(M[n] == 1)[0]
    return aoi[n, subs].mean()

for t in range(T):

    aoi[M == 1] += 1.0

    start = t % N
    order = [(start + i) % N for i in range(N)]

    candidates = []
    scores = []

    for n in order:
        s = topic_mean_aoi(n)
        if s > lam:
            candidates.append(n)
            scores.append(s)

    chosen = candidates

    for n in chosen:
        subs = np.where(M[n] == 1)[0]
        success = np.random.rand(subs.size) <= q
        aoi[n, subs[success]] = 1.0

    tx = len(chosen)
    tx_hist[t] = tx

    lam = max(0.0, beta * lam + (1.0 - beta) * (tx - Cbar))
    lam_hist[t] = lam

    avg_aoi_hist[t] = aoi[M == 1].mean()

print("Final lambda:", lam_hist[-1])
print("Average transmissions per slot:", tx_hist.mean())
print("Average AoI:", avg_aoi_hist.mean())

plt.figure()
plt.plot(lam_hist)
plt.xlabel("Time slot")
plt.ylabel("lambda")
plt.tight_layout()
plt.savefig("lambda_update.png", dpi=300)
plt.show()

plt.figure()
plt.plot(tx_hist)
plt.xlabel("Time slot")
plt.ylabel("transmissions")
plt.tight_layout()
plt.savefig("tx_per_slot.png", dpi=300)
plt.show()

plt.figure()
plt.plot(avg_aoi_hist)
plt.xlabel("Time slot")
plt.ylabel("Average AoI")
plt.tight_layout()
plt.savefig("avg_aoi_price_capacity.png", dpi=300)
plt.show()