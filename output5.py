import matplotlib.pyplot as plt
import numpy as np
import math

def load2D4list(dirs):
    r = np.load(dirs)
    return r["arr_0"],r["arr_1"],r["arr_2"],r["arr_3"]


rls = [0.0001,0.000075,0.00005]
kends = [1.,1.25,1.5,1.75,2.,2.25,2.5,2.75,3]
kx = [2,2.5,3,3.5,4,4.5,5,5.5,6]
colors  = ['blue','red','green']
labels = ['learning rate = 0.0001','learning rate = 0.000075','learning rate = 0.00005']
count = 0
plt.figure(figsize=(7, 5))
plt.title('Results')
plt.xlabel('final k value')
plt.ylabel('mean of rewards')
for rl in rls:
    
    mean_r = []
    s = ''
    if rl==0.0001:
        s = '1'
    if rl==0.000075:
        s = '075'
    if rl==0.000050:
        s = '050'
    for k_end in kends:
        str = "data/analysis/CartPolerl%s_kend%f.npz"%(s,k_end*2.)       
        mean_losses,var_losses,mean_rewards0,var_rewards0 = load2D4list(str)
        frame0 = [i for i in range(mean_rewards0.size)]
        a = np.mean(mean_rewards0[-50:-1])
        mean_r.append(a)
    
    my_y_ticks = np.arange(50, 200, 30)   
    

    plt.plot(kx, mean_r, color=colors[count], label=labels[count])
    plt.yticks(my_y_ticks)
    count += 1


plt.legend()
plt.show()
        