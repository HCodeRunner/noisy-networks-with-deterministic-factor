import matplotlib.pyplot as plt
import numpy as np
import math

def load2D4list(dirs):
    r = np.load(dirs)
    return r["arr_0"],r["arr_1"],r["arr_2"],r["arr_3"],r["arr_4"],r["arr_5"],r["arr_6"]





def load2D4list2(dirs):
    r = np.load(dirs)
    return r["arr_0"],r["arr_1"],r["arr_2"],r["arr_3"]
mean_losses, var_losses, mean_rewards, var_rewards=load2D4list2('data/improvednoisy_pong.npz')
frame = [i for i in range(mean_rewards.size)]
#plt.plot(frame, mean_losses[100001:])
plt.figure(figsize=(15, 4))
ax1 = plt.subplot(1,3,1)
plt.sca(ax1)
plt.plot(frame, mean_rewards)
plt.ylabel('rewards')
plt.xlabel('episodes')
plt.title('learning rate = 0.0001\nfinal_k = 2')
plt.tight_layout()


mean_losses, var_losses, mean_rewards, var_rewards, weight_sigmas1, bais_sigmas1, frame_list1=load2D4list('data/improvednoisy_pong1.npz')
frame = [i for i in range(mean_rewards.size)]
#plt.plot(frame, mean_losses[100001:])

ax2 = plt.subplot(1,3,2)
plt.sca(ax2)
plt.plot(frame, mean_rewards)
plt.ylabel('rewards')
plt.xlabel('episodes')
plt.title('learning rate = 0.0001\nfinal_k = 4')
plt.tight_layout()


mean_losses, var_losses, mean_rewards, var_rewards, weight_sigmas, bais_sigmas, frame_list=load2D4list('data/improvednoisy_pong2.npz')
frame = [i for i in range(mean_rewards.size)]


ax3 = plt.subplot(1,3,3)
plt.sca(ax3)
plt.plot(frame, mean_rewards)
plt.ylabel('rewards')
plt.xlabel('episodes')
plt.title('learning rate = 0.0001\nfinal_k = 6')
plt.tight_layout()


plt.show()