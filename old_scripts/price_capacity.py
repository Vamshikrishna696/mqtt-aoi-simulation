import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

N, K, T = 7, 5, 300
q = 0.9
Cbar = 3
beta = 0.94
lam = 0.0

M = (np.random.rand(N, K) < 0.5).astype(int)
for n in range(N):
    if M[n].sum() == 0:
        M[n, np.random.randint(0, K)] = 1

aoi = np.zeros((N, K), dtype=float)
aoi[M == 1] = 1.0

lam_hist = np.zeros(T)
avg_aoi_hist = np.zeros(T)
tx_hist = np.zeros(T)

for t in range(T):
    aoi[M == 1] += 1.0

    start = t % N
    order = [(start + i) % N for i in range(N)]

    scores = np.zeros(N)
    for n in order:
        subs = np.where(M[n] == 1)[0]
        scores[n] = aoi[n, subs].mean()

    candidates = [n for n in order if scores[n] > lam]

    if len(candidates) > Cbar:
        candidates_sorted = sorted(candidates, key=lambda n: scores[n], reverse=True)
        chosen = candidates_sorted[:Cbar]
    else:
        chosen = candidates

    for n in chosen:
        subs = np.where(M[n] == 1)[0]
        success = (np.random.rand(subs.size) <= q)
        aoi[n, subs[success]] = 1.0

    tx = len(chosen)
    tx_hist[t] = tx

    lam = max(0.0, beta * lam + (1.0 - beta) * (tx - Cbar))
    lam_hist[t] = lam

    avg_aoi_hist[t] = aoi[M == 1].mean()

print("MODE: price + hard capacity cap")
print("Final lambda:", lam_hist[-1])
print("Average transmissions per slot:", tx_hist.mean())
print("Max transmissions in a slot:", tx_hist.max())
print("Average AoI (active pairs):", avg_aoi_hist.mean())

plt.figure()
plt.plot(lam_hist)
plt.xlabel("Time slot")
plt.ylabel("lambda")
plt.title("Lambda (price + cap)")
plt.tight_layout()
plt.savefig("lambda_price_cap.png", dpi=300)
plt.show()

plt.figure()
plt.plot(tx_hist)
plt.xlabel("Time slot")
plt.ylabel("transmissions")
plt.title("Tx per slot (price + cap)")
plt.tight_layout()
plt.savefig("tx_price_cap.png", dpi=300)
plt.show()

plt.figure()
plt.plot(avg_aoi_hist)
plt.xlabel("Time slot")
plt.ylabel("Average AoI")
plt.title("Average AoI (price + cap)")
plt.tight_layout()
plt.savefig("avg_aoi_price_cap.png", dpi=300)
plt.show()