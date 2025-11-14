import os
os.makedirs("plots", exist_ok=True)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
from main import train_dqn

def exp(batch, update, episodes=20):
    scores, rewards = train_dqn(episodes=episodes, batch_size=batch, update_rate=update)
    return scores, rewards

print("\nRunning batch size experiments...")

s8, r8 = exp(8, 10)
s16, r16 = exp(16, 10)

# BATCH SIZE - SCORE COMPARISON
plt.figure()
plt.plot(s8, label="Batch Size = 8")
plt.plot(s16, label="Batch Size = 16")
plt.legend()
plt.title("Batch Size Comparison - Score")
plt.savefig("plots/batch_size_score_comparison.png")
plt.close()

# BATCH SIZE - AVG REWARD COMPARISON
plt.figure()
plt.plot(r8, label="Batch Size = 8")
plt.plot(r16, label="Batch Size = 16")
plt.legend()
plt.title("Batch Size Comparison - Average Reward")
plt.savefig("plots/batch_size_reward_comparison.png")
plt.close()


print("\nRunning target update experiments...")

s3, r3 = exp(8, 3)
s10, r10 = exp(8, 10)

# UPDATE RATE - SCORE
plt.figure()
plt.plot(s3, label="Update Rate = 3")
plt.plot(s10, label="Update Rate = 10")
plt.legend()
plt.title("Target Network Update Rate - Score")
plt.savefig("plots/update_rate_score_comparison.png")
plt.close()

# UPDATE RATE - AVG REWARD
plt.figure()
plt.plot(r3, label="Update Rate = 3")
plt.plot(r10, label="Update Rate = 10")
plt.legend()
plt.title("Target Network Update Rate - Average Reward")
plt.savefig("plots/update_rate_reward_comparison.png")
plt.close()

print("All experiments completed and plots saved successfully.")
