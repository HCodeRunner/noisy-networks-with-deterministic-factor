import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def load2D4list(dirs):
    r = np.load(dirs)
    return r["arr_0"],r["arr_1"],r["arr_2"],r["arr_3"]

mean_losses,var_losses,mean_rewards,var_rewards = load2D4list('data/improvednoisy_pong1_five.npz')
mean_losses,var_losses,mean_rewards1,var_rewards1 = load2D4list('data/noisydqn_pong_five1.npz')
frame = [i for i in range(mean_rewards.size)]
frame1 = [i for i in range(mean_rewards1.size)]
plt.title('Results')
plt.plot(frame1, mean_rewards1,color='blue', label='noisy netwoks DQN')
plt.plot(frame, mean_rewards,color='red', label='improved noisy netwoks DQN')
plt.legend()
plt.xlabel('episodes')
plt.ylabel('mean of rewards')
plt.legend()
plt.show()