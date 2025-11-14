import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

import gym
import numpy as np
import torch
import matplotlib.pyplot as plt

from assignment3_utils import process_frame
from dqn_agent_final import DQNAgent

import os
os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

IMAGE_SHAPE = (84, 80)

# --------------------------------------------------------
# FRAME STACKING (FINAL FIXED VERSION)
# --------------------------------------------------------
def stack_frames(stacked, new_frame, new_episode):
    frame = process_frame(new_frame, IMAGE_SHAPE)  # (1,84,80,1)
    frame = frame.squeeze()                       # (84,80)

    if new_episode or stacked is None:
        stacked = np.stack([frame] * 4, axis=0)   # (4,84,80)
    else:
        stacked = np.concatenate([stacked[1:], frame[np.newaxis, ...]], axis=0)

    return stacked


# --------------------------------------------------------
# TRAINING LOOP
# --------------------------------------------------------
def train_dqn(episodes=50, batch_size=8, update_rate=10):
    env = gym.make("PongDeterministic-v4", render_mode=None)
    agent = DQNAgent(env.action_space.n)

    scores = []
    avg_rewards = []

    for ep in range(episodes):

        # SAFE RESET
        reset_output = env.reset()
        obs = reset_output[0] if isinstance(reset_output, tuple) else reset_output

        state = stack_frames(None, obs, True)
        score = 0
        done = False

        while not done:
            action = agent.choose_action(state.reshape(1, 4, 84, 80))

            # SAFE STEP
            step_output = env.step(action)
            if len(step_output) == 5:
                next_obs, reward, term, trunc, info = step_output
                done = term or trunc
            else:
                next_obs, reward, done, info = step_output

            next_state = stack_frames(state, next_obs, False)
            clipped_reward = np.sign(reward)

            agent.memory.add(state, action, clipped_reward, next_state, done)
            agent.train_step(batch_size)

            state = next_state
            score += clipped_reward

        scores.append(score)
        avg_rewards.append(np.mean(scores[-5:]))

        print(f"Episode {ep+1}/{episodes} | Score={score} | Avg5={avg_rewards[-1]:.2f}")

        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        if (ep + 1) % update_rate == 0:
            agent.update_target()

    # SAVE MODEL
    torch.save(agent.policy_net.state_dict(), "models/pong_dqn.pth")

    # SAVE TRAINING PLOTS
    plt.figure()
    plt.plot(scores)
    plt.title("Score per Episode")
    plt.savefig("plots/score_per_episode.png")
    plt.close()

    plt.figure()
    plt.plot(avg_rewards)
    plt.title("Average Reward (Last 5)")
    plt.savefig("plots/avg_reward_plot.png")
    plt.close()

    env.close()
    return scores, avg_rewards
