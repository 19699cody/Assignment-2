import gymnasium as gym
import numpy as np
import imageio.v2 as imageio

# Load the Q-table produced by training
q_table = np.load("q_table.npy")

# Create the Taxi environment with video rendering enabled
env = gym.make("Taxi-v4", render_mode="rgb_array")

state, info = env.reset(seed=42)
frames = [env.render()]

total_reward = 0
max_steps = 200

for step in range(max_steps):
    # Choose the best action from the trained Q-table
    action = int(np.argmax(q_table[state]))

    state, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    frames.append(env.render())

    if terminated or truncated:
        break

env.close()

# Save the recorded episode as an MP4
with imageio.get_writer(
    "taxi_agent.mp4",
    fps=2,
    codec="libx264"
) as video:
    for frame in frames:
        video.append_data(frame)

print(f"Video saved as taxi_agent.mp4")
print(f"Steps: {step + 1}")
print(f"Total reward: {total_reward}")
print(f"Successful drop-off: {terminated}")
