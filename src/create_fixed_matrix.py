"""
Create and save one fixed subscription matrix for all policies.
"""

from config import N, K, SUBSCRIPTION_PROB, SEED
from src.utils import generate_fixed_subscription_matrix, save_matrix


def main():
    matrix = generate_fixed_subscription_matrix(
        n=N,
        k=K,
        sub_prob=SUBSCRIPTION_PROB,
        seed=SEED
    )

    save_matrix(matrix, "data/fixed_subscription_matrix.npy")

    print("Fixed subscription matrix created successfully.")
    print(matrix)


if __name__ == "__main__":
    main()