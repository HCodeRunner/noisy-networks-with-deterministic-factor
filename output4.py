import matplotlib.pyplot as plt
import numpy as np
import math

def load2D4list(dirs):
    r = np.load(dirs)
    return r["arr_0"],r["arr_1"],r["arr_2"],r["arr_3"]

rls = [0.0001,0.000075,0.00005]
kends = [1.,1.25,1.5,1.75,2.,2.25,2.5,2.75,3]
count = 0
plt.figure(figsize=(15, 16))
for rl in rls:
    for k_end in kends:
        count += 1
        s = ''
        if rl==0.0001:
            s = '1'
        if rl==0.000075:
            s = '075'
        if rl==0.000050:
            s = '050'
        str = "data/analysis/CartPolerl%s_kend%f.npz"%(s,k_end*2.)
        title = "learning rate = %f\nfinal_k = %f"%(rl,k_end*2.)
        mean_losses,var_losses,mean_rewards0,var_rewards0 = load2D4list(str)
        frame0 = [i for i in range(mean_rewards0.size)]
        ax1 = plt.subplot(7,4,count)
        plt.sca(ax1)
        if count%4 ==1:
            plt.ylabel('rewards')
        if count > 23:
            plt.xlabel('episodes')
        plt.title(title)
        plt.plot(frame0,mean_rewards0,color='green')
        var_rewards0 = [math.sqrt(a) for a in var_rewards0]
        up = mean_rewards0+var_rewards0
        up = np.where(up < 200, up, 200)

        low = mean_rewards0-var_rewards0
        low = np.where(low > 0, low, 0)
        #a = 0
        #b = mean_rewards0.size
        plt.fill_between(frame0,up,low,color='blue',alpha=0.25)
        plt.tight_layout()
        
        

plt.show()
        