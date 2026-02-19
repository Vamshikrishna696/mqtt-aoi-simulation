import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

N = 7
K = 5
T = 200
B = 3
p_success = 0.9

M = (np.random.rand(N, K) < 0.5).astype(int)

active_pairs = np.argwhere(M == 1)
if active_pairs.size == 0:
    raise RuntimeError("No active subscriptions in M. Increase subscription probability.")

track_pub, track_sub = active_pairs[0]  # pick first active pair (deterministic)
print("Publisher–Subscriber matrix M (rows=publishers, cols=subscribers):")
print(M)
print(f"Tracking Publisher {track_pub+1} -> Subscriber {track_sub+1} (M[{track_pub},{track_sub}] = 1)")

aoi = np.ones((N, K), dtype=float)

aoi_track = np.empty(T, dtype=float)
avg_aoi_active = np.empty(T, dtype=float)

for t in range(T):
    aoi += 1

    start = (t * B) % N
    chosen_pubs = [(start + i) % N for i in range(B)]

    for pub in chosen_pubs:
        subs = np.where(M[pub] == 1)[0]
        if subs.size == 0:
            continue
        success = (np.random.rand(subs.size) <= p_success)
        aoi[pub, subs[success]] = 1

    aoi_track[t] = aoi[track_pub, track_sub]
    avg_aoi_active[t] = np.mean(aoi[M == 1])

print("\nAverage AoI (active pairs):", float(avg_aoi_active.mean()))
print(f"Average AoI (tracked pair P{track_pub+1}-S{track_sub+1}):", float(aoi_track.mean()))

plt.figure()
plt.plot(aoi_track)
plt.xlabel("Time slot")
plt.ylabel("AoI")
plt.title(f"AoI of Publisher {track_pub+1} at Subscriber {track_sub+1} (Matrix RR)")
plt.grid(True)
plt.tight_layout()
plt.savefig("aoi_tracked_pair_matrix_rr.png", dpi=300)
plt.show()

plt.figure()
plt.plot(avg_aoi_active)
plt.xlabel("Time slot")
plt.ylabel("Average AoI (active pairs)")
plt.title("Average AoI over Active Pairs (Matrix RR)")
plt.grid(True)
plt.tight_layout()
plt.savefig("avg_aoi_active_pairs_matrix_rr.png", dpi=300)
plt.show()
