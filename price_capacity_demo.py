import numpy as np
import matplotlib.pyplot as plt


def compute_score(aoi_value: int, success_prob: float, lam: float) -> float:
    return aoi_value * success_prob - lam


def main() -> None:
    np.random.seed(1)

    n_topics = 7
    capacity_c = 3  # not enforced for this baseline
    t_slots = 20

    p = np.full(n_topics, 0.9, dtype=float)
    lam = 0.3

    aoi = np.ones(n_topics, dtype=int)

    aoi_log = np.zeros((t_slots, n_topics), dtype=int)
    score_log = np.zeros((t_slots, n_topics), dtype=float)
    act_log = np.zeros((t_slots, n_topics), dtype=int)

    for t in range(t_slots):
        aoi += 1

        scores = np.array([compute_score(aoi[i], p[i], lam) for i in range(n_topics)], dtype=float)
        chosen = np.where(scores > 0.0)[0]

        act_log[t, chosen] = 1

        for i in chosen:
            if np.random.rand() <= p[i]:
                aoi[i] = 1

        aoi_log[t] = aoi
        score_log[t] = scores

    np.set_printoptions(suppress=True)

    print("\nPARAMETERS")
    print(f"N={n_topics}, C={capacity_c} (not enforced), T={t_slots}, p={p[0]:.2f}, lambda={lam:.2f}")

    print("\nAoI matrix (rows=time, cols=topics 1..N)")
    print(aoi_log)

    print("\nScore/price matrix")
    print(np.round(score_log, 3))

    print("\nAction matrix (1=attempt update, 0=idle)")
    print(act_log)

    avg_aoi_per_topic = aoi_log.mean(axis=0)
    avg_aoi_all = aoi_log.mean()

    print("\nAverage AoI per topic")
    print(np.round(avg_aoi_per_topic, 3))

    print(f"\nOverall average AoI = {avg_aoi_all:.3f}")

    updates_per_slot = act_log.sum(axis=1)
    print("\nUpdates attempted per slot (can exceed C in this baseline)")
    print(updates_per_slot)

    plt.figure()
    plt.plot(aoi_log[:, 6])
    plt.xlabel("Time slot")
    plt.ylabel("AoI")
    plt.title("AoI of Topic 7")
    plt.grid(True)
    plt.savefig("topic7_aoi.png", dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure()
    plt.plot(aoi_log.mean(axis=1))
    plt.xlabel("Time slot")
    plt.ylabel("Average AoI")
    plt.title("Average AoI (all topics)")
    plt.grid(True)
    plt.savefig("avg_aoi.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
