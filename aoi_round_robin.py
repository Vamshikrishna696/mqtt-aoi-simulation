import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)


N = 7
K = 3
T = 200

p = 0.9 * np.ones(N)
lambda_cost = 0.3

aoi = np.ones(N)

aoi_topic7 = []
avg_aoi = []

for t in range(T):
    start = (t * K) % N
    selected = [(start + i) % N for i in range(K)]

    aoi += 1

    for i in selected:
        if np.random.rand() <= p[i]:
            aoi[i] = 1

    aoi_topic7.append(aoi[6])
    avg_aoi.append(np.mean(aoi))

print("Average AoI of Topic 7:", np.mean(aoi_topic7))
print("Average AoI of all topics:", np.mean(avg_aoi))

plt.plot(aoi_topic7)
plt.xlabel("Time Slot")
plt.ylabel("AoI")
plt.title("AoI of Topic 7 under Round Robin Scheduling")
plt.grid(True)
plt.savefig("aoi_topic7_round_robin.png", dpi=300, bbox_inches="tight")
plt.show()

