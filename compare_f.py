import numpy as np
import matplotlib.pyplot as plt

decay_rates = [0.99, 0.999, 0.9999]

plt.figure(figsize=(10, 6))

for decay in decay_rates:
    rewards = np.load(f"question_f_rewards_decay_{decay}.npy")

    # Calculate the same 100-episode moving average
    moving_average = np.convolve(
        rewards, np.ones(100) / 100, mode="valid"
    )

    plt.plot(
        np.arange(100, len(rewards) + 1),
        moving_average,
        label=f"Epsilon decay = {decay}"
    )

plt.xlabel("Episode")
plt.ylabel("Average Reward (100-episode moving average)")
plt.title("Effect of Epsilon Decay on Q-Learning")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("question_f_comparison.png", dpi=300)
plt.show()
