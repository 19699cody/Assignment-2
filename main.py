import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# QUESTION B: Q-LEARNING AGENT IMPLEMENTATION
# ============================================================

# Hyperparameters used for baseline training.
# These values can be changed later for the experiments.
EPISODES = 10_000
MAX_STEPS = 200

ALPHA = 0.1          # Learning rate
GAMMA = 0.99         # Discount factor
EPSILON = 1.0        # Initial exploration probability
EPSILON_MIN = 0.01   # Minimum exploration probability
EPSILON_DECAY = 0.99


def train_q_learning(
    env,
    episodes=EPISODES,
    max_steps=MAX_STEPS,
    alpha=ALPHA,
    gamma=GAMMA,
    epsilon=EPSILON,
    epsilon_min=EPSILON_MIN,
    epsilon_decay=EPSILON_DECAY,
    seed=42,
):
    """Train a tabular Q-learning agent."""

    rng = np.random.default_rng(seed)

    # QUESTION B: Q-VALUE INITIALIZATION
    # Create a Q-table with one row per state and one column
    # per action. All Q-values initially start at zero.
    # Taxi-v4 has 500 states and 6 actions.
    q_table = np.zeros(
        (env.observation_space.n, env.action_space.n),
        dtype=np.float64,
    )

    episode_rewards = []

    # QUESTION B: TRAINING LOOP
    # Repeat the learning process over multiple episodes.
    # Each episode begins with a reset of the environment.
    for episode in range(episodes):
        state, info = env.reset(seed=seed + episode)
        total_reward = 0

        for step in range(max_steps):
            # QUESTION B: EPSILON-GREEDY ACTION SELECTION
            # Explore by choosing a random action with
            # probability epsilon. Otherwise, exploit by
            # choosing the action with the highest Q-value.
            if rng.random() < epsilon:
                action = int(rng.integers(env.action_space.n))
            else:
                action = int(np.argmax(q_table[state]))

            # Perform the selected action and observe the result.
            next_state, reward, terminated, truncated, info = (
                env.step(action)
            )

            # QUESTION B: Q-LEARNING UPDATE
            # If the task has terminated, use only the reward.
            # Otherwise, include the best estimated future value.
            # A time-limit truncation can still use the next
            # state's estimated value in the update.
            if terminated:
                target = reward
            else:
                target = reward + gamma * np.max(q_table[next_state])

            # Q(s,a) <- Q(s,a) + alpha * (target - Q(s,a))
            q_table[state, action] += alpha * (
                target - q_table[state, action]
            )

            state = next_state
            total_reward += reward

            # End the episode after success or truncation.
            if terminated or truncated:
                break

        episode_rewards.append(total_reward)

        # Reduce exploration after each episode, while keeping
        # a small minimum probability of trying random actions.
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Print progress every 1,000 episodes.
        if (episode + 1) % 1000 == 0:
            recent_mean = np.mean(episode_rewards[-1000:])
            print(
                f"Episode {episode + 1:5d}/{episodes} | "
                f"Recent mean reward: {recent_mean:7.2f} | "
                f"Epsilon: {epsilon:.3f}"
            )

    return q_table, episode_rewards


def evaluate_agent(env, q_table, episodes=100, seed=100_000):
    """Evaluate the learned policy without intentional exploration."""

    rewards = []
    steps_per_episode = []
    successes = 0

    for episode in range(episodes):
        state, info = env.reset(seed=seed + episode)
        total_reward = 0

        for step in range(MAX_STEPS):
            # Evaluation uses the best-known action rather than
            # randomly exploring.
            action = int(np.argmax(q_table[state]))

            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated:
                successes += 1

            if terminated or truncated:
                break

        rewards.append(total_reward)
        steps_per_episode.append(step + 1)

    print("\nGreedy policy evaluation")
    print(f"Episodes: {episodes}")
    print(f"Mean reward: {np.mean(rewards):.2f}")
    print(f"Success rate: {successes / episodes * 100:.1f}%")
    print(f"Mean steps per episode: {np.mean(steps_per_episode):.2f}")

    return rewards, steps_per_episode


def main():
    # Create the standard Taxi-v4 environment.
    env = gym.make("Taxi-v4")

    print("Training Q-learning agent...")
    q_table, rewards = train_q_learning(env)

    # Save the learned Q-table so it can be reused.
    np.save("q_table.npy", q_table)

    # QUESTION F: SAVE REWARDS FOR THE COMPARISON PLOT
    # Each epsilon decay rate gets its own file.
    reward_filename = (
        f"question_f_rewards_decay_{EPSILON_DECAY}.npy"
    )
    np.save(reward_filename, rewards)

    # QUESTION C: BASELINE TRAINING PLOT
    # Plot episode rewards and their 100-episode moving average.
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.25, label="Episode reward")

    window = 100
    if len(rewards) >= window:
        moving_average = np.convolve(
            rewards,
            np.ones(window) / window,
            mode="valid",
        )
        plt.plot(
            range(window - 1, len(rewards)),
            moving_average,
            label=f"{window}-episode moving average",
        )

    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("Taxi Q-learning: Baseline Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig("baseline_training.png", dpi=150)
    plt.show()

    # QUESTION G: BASELINE AGENT EVALUATION
    # Evaluate the trained agent over 100 episodes.
    evaluate_agent(env, q_table)

    env.close()

    print("\nSaved Q-table to q_table.npy")
    print("Saved training plot to baseline_training.png")
    print(f"Saved episode rewards to {reward_filename}")


if __name__ == "__main__":
    main()
