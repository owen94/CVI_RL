import numpy as np

import matplotlib.pyplot as plt

### result in CVI + NOISE

path1 = '../Results/cvi/result/LunarLanderContinuous-v2/NOISE+CVI/noise_10000.0/lr_10.0/beta_0.1/reward.npy'
path2 = '../Results/cvi/result/LunarLanderContinuous-v2/NOISE+CVI/noise_10000.0/lr_10.0/beta_0.01/reward.npy'
path3 = '../Results/cvi/result/LunarLanderContinuous-v2/NOISE+CVI/noise_10000.0/lr_100.0/beta_0.1/reward.npy'
path4 = "../Results/cvi/result/LunarLanderContinuous-v2/NOISE+CVI/noise_10000.0/lr_100.0/beta_0.01/reward.npy"

path5 = '../Results/cvi/result/LunarLanderContinuous-v2/NOISE+CVI/noise_100000.0/lr_10.0/beta_0.1/reward.npy'
path6 = '../Results/cvi/result/LunarLanderContinuous-v2/NOISE+CVI/noise_100000.0/lr_10.0/beta_0.01/reward.npy'
path7 = '../Results/cvi/result/LunarLanderContinuous-v2/NOISE+CVI/noise_100000.0/lr_100.0/beta_0.1/reward.npy'
path8 = "../Results/cvi/result/LunarLanderContinuous-v2/NOISE+CVI/noise_100000.0/lr_100.0/beta_0.01/reward.npy"

### result in SGD + NOISE

s_path1 = '../Results/cvi/result/LunarLanderContinuous-v2/NOISE+SGD/noise_10000.0/lr_0.0001/beta_0.1/reward.npy'
s_path2 = '../Results/cvi/result/LunarLanderContinuous-v2/NOISE+SGD/noise_10000.0/lr_0.0001/beta_0.01/reward.npy'
s_path3 = '../Results/cvi/result/LunarLanderContinuous-v2/NOISE+SGD/noise_10000.0/lr_0.0001/beta_0.001/reward.npy'
s_path4 = '../Results/cvi/result/LunarLanderContinuous-v2/NOISE+SGD/noise_10000.0/lr_0.0001/beta_0.0001/reward.npy'


### result in Adam + NOISE

a_path1 = '../Results/cvi/result/LunarLanderContinuous-v2/NOISE+ADAM/noise_10000.0/lr_0.0001/beta_0.1/reward.npy'
a_path2 = '../Results/cvi/result/LunarLanderContinuous-v2/NOISE+ADAM/noise_10000.0/lr_0.0001/beta_0.01/reward.npy'
a_path3 = '../Results/cvi/result/LunarLanderContinuous-v2/NOISE+ADAM/noise_10000.0/lr_0.0001/beta_0.001/reward.npy'
a_path4 = '../Results/cvi/result/LunarLanderContinuous-v2/NOISE+ADAM/noise_10000.0/lr_0.0001/beta_0.0001/reward.npy'

### result in SGD + CONSTANT + NOISE

sc_path1 = '../Results/cvi/result/LunarLanderContinuous-v2/NOISE+SGD+CONS/noise_10000.0/lr_0.0001/beta_0.1/reward.npy'
sc_path2 = '../Results/cvi/result/LunarLanderContinuous-v2/NOISE+SGD+CONS/noise_100000.0/lr_0.0001/beta_0.1/reward.npy'

# result from pure SGD

sn_path1 = '../Results/cvi/result/LunarLanderContinuous-v2/SGD/noise_100000/lr_0.0001/beta_0.1/reward.npy'


# result from pure ADAM
an_path1 = '../Results/cvi/result/LunarLanderContinuous-v2/ADAM/noise_100000/lr_0.0001/beta_0.1/reward.npy'


plot_len = 500
x = np.arange(plot_len)

# load CVI results  the best is c3
c1 = np.load(path1)[:plot_len]
c2 = np.load(path2)[:plot_len]
c3 = np.load(path3)[:plot_len]
c4 = np.load(path4)[:plot_len]

c5 = np.load(path5)[:plot_len]
c6 = np.load(path6)[:plot_len]
c7 = np.load(path7)[:plot_len]
c8 = np.load(path8)[:plot_len]

# load SGD + NOISE results  $ The best is s3
s1 = np.load(s_path1)[:plot_len]
s2 = np.load(s_path2)[:plot_len]
s3 = np.load(s_path3)[:plot_len]
s4 = np.load(s_path4)[:plot_len]

# load SGD + ADAM  best is a2
a1 = np.load(a_path1)[:plot_len]
a2 = np.load(a_path2)[:plot_len]
a3 = np.load(a_path3)[:plot_len]
a4 = np.load(a_path4)[:plot_len]

# choose sc1 for fair comparison
sc1 = np.load(sc_path1)[:plot_len]
sc2 = np.load(sc_path2)[:plot_len]

#
sn1 = np.load(sn_path1)[:plot_len]

an1 = np.load(an_path1)[:plot_len]

plt.plot(x,a2)
plt.plot(x,s3)
plt.plot(x,c3)
#plt.plot(x,sc1)
plt.plot(x, sn1 ,'m')
plt.plot(x, an1)

plt.legend([ 'ADAM+NOISE', 'SGD+NOISE', 'CVI+NOISE','SGD+NoNOISE', 'ADAM+NoNOISE'])
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('Training rewards of different methods')
#plt.legend([ 'ADAM+NOISE', 'SGD+NOISE', 'CVI+NOISE', 'SGD+CONS+NOISE','SGD', 'ADAM'])
plt.show()