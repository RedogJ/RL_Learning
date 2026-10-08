import gymnasium as gym


env = gym.make("CartPole-v1")

state, info = env.reset()

total_reward = 0
step = 0

while True:
    action = env.action_space.sample()

    next_state, reward, terminated, truncated, info = env.step(action)

    print(f"step: {step}")
    print(f"state: {state}")
    print(f"action: {action}")
    print(f"reward: {reward}")
    print(f"next_state: {next_state}")
    print(f"terminated: {terminated}")
    print(f"truncated: {truncated}")
    print("-" * 50)

    total_reward += reward

    state = next_state
    step += 1

    if terminated or truncated:
        break

print("Episode finished!")
print("Total reward:", total_reward)
print("Total steps:", step)

env.close()