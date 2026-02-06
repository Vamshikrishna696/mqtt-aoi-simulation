import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

N = 7
K = 3
T = 200

p = 0.9 * np.ones(N)
lam = 0.8

aoi = np.ones(N)

aoi_history = []
avg_aoi_history = []
u_history = []

for t in range(T):
    start = (t * K) % N
    selected = [(start + i) % N for i in range(K)]

    aoi += 1
    u = np.zeros(N)

    for i in selected:
        if p[i] * aoi[i] > lam:
            u[i] = 1
            if np.random.rand() <= p[i]:
                aoi[i] = 1

    u_history.append(u)
    aoi_history.append(aoi.copy())
    avg_aoi_history.append(np.mean(aoi))

aoi_history = np.array(aoi_history)
u_history = np.array(u_history)

print("Average AoI of Topic 7:", np.mean(aoi_history[:, 6]))
print("Average AoI of all topics:", np.mean(avg_aoi_history))
print("Average transmissions per slot:", np.mean(np.sum(u_history, axis=1)))

plt.plot(aoi_history[:, 6])
plt.xlabel("Time slot")
plt.ylabel("AoI")
plt.title("AoI of Topic 7 (Round Robin with Price)")
plt.grid(True)
plt.savefig("topic7_aoi_price.png", dpi=300, bbox_inches="tight")
plt.show()

plt.plot(avg_aoi_history)
plt.xlabel("Time slot")
plt.ylabel("Average AoI")
plt.title("Average AoI of All Topics")
plt.grid(True)
plt.savefig("avg_aoi_price.png", dpi=300, bbox_inches="tight")
plt.show()
