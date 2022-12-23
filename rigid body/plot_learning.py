from math import pi
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.ndimage import gaussian_filter
formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))
import numpy as np
plt.style.use('seaborn-whitegrid')
matplotlib.rc('legend', frameon=True, fontsize='medium', loc='upper left', framealpha=0.6)#
matplotlib.rc('text', usetex=True)
matplotlib.rcParams["text.latex.preamble"]=[r"\usepackage{times}", r"\usepackage{amsmath}"]
matplotlib.rc('font', family='serif', serif='CMU Serif', monospace='Computer Modern Typewriter', size=14)


figsize = (8,6.4)# (4,3.2) #



plt.figure(figsize=figsize)
ax=plt.gca()
plt.xlabel(r"simulation time ($T$)")
plt.ylabel(r"$E$")

f=open("default_init/xpInput_lm0.04_ga0.011.txt",'r')
lines=f.readlines()
xp_x=np.array([float(line.split(',')[0]) for line in lines])
xp_y=np.array([float(line.split(',')[1]) for line in lines])
f.close()
plt.plot(xp_x,gaussian_filter(xp_y, sigma=40, mode="nearest"), alpha=0.6, linewidth=1, label="1")
f=open("default_init/xpInput_lm0.04_ga0.012.txt",'r')
lines=f.readlines()
xp_x=np.array([float(line.split(',')[0]) for line in lines])
xp_y=np.array([float(line.split(',')[1]) for line in lines])
f.close()
plt.plot(xp_x,gaussian_filter(xp_y, sigma=40, mode="nearest"), alpha=0.6,linewidth=1, label="2")

f=open("default_init/xpInput_lm0.04_ga0.013.txt",'r')
lines=f.readlines()
xp_x=np.array([float(line.split(',')[0]) for line in lines])
xp_y=np.array([float(line.split(',')[1]) for line in lines])
f.close()
plt.plot(xp_x,gaussian_filter(xp_y, sigma=40, mode="nearest"), alpha=0.6,linewidth=1, label="3")

f=open("default_init/xpInput_lm0.04_ga0.014.txt",'r')
lines=f.readlines()
xp_x=np.array([float(line.split(',')[0]) for line in lines])
xp_y=np.array([float(line.split(',')[1]) for line in lines])
f.close()
plt.plot(xp_x,gaussian_filter(xp_y, sigma=40, mode="nearest"), alpha=0.6,linewidth=1, label="4")

f=open("default_init/xpInput_lm0.04_ga0.015.txt",'r')
lines=f.readlines()
xp_x=np.array([float(line.split(',')[0]) for line in lines])
xp_y=np.array([float(line.split(',')[1]) for line in lines])
f.close()
plt.plot(xp_x,gaussian_filter(xp_y, sigma=40, mode="nearest"), alpha=0.6,linewidth=1, label="5")


#ax.set_xticks([0,250000,500000])
#ax.set_xticklabels([r"$0$",r"$2.5\times10^5$",r"$5\times10^5$"])
#ax.set_yticks(yticks)
#ax.set_yticklabels(["{:.3f}".format(f) for f in yticks])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

ax.set_ylim(bottom=0.)#top=0.4,
#ax.set_xlim(left=0.,right=549444+10000)
#plt.legend(loc=1)
plt.title("DQN")
plt.tight_layout()
plt.subplots_adjust(left=0.17,bottom=0.155,right=0.975,top=0.9)
plt.savefig("default_init/test_DQN.png")
plt.clf()





plt.figure(figsize=figsize)
ax=plt.gca()
plt.xlabel(r"simulation time ($T$)")
plt.ylabel(r"$E$")

f=open("default_init/xpInput_lm0.04_ga0.011CDQN.txt",'r')
lines=f.readlines()
xp_x=np.array([float(line.split(',')[0]) for line in lines])
xp_y=np.array([float(line.split(',')[1]) for line in lines])
f.close()
plt.plot(xp_x,gaussian_filter(xp_y, sigma=40, mode="nearest"), alpha=0.6, linewidth=1, label="1")
f=open("default_init/xpInput_lm0.04_ga0.012CDQN.txt",'r')
lines=f.readlines()
xp_x=np.array([float(line.split(',')[0]) for line in lines])
xp_y=np.array([float(line.split(',')[1]) for line in lines])
f.close()
plt.plot(xp_x,gaussian_filter(xp_y, sigma=40, mode="nearest"), alpha=0.6, linewidth=1, label="2")

f=open("default_init/xpInput_lm0.04_ga0.013CDQN.txt",'r')
lines=f.readlines()
xp_x=np.array([float(line.split(',')[0]) for line in lines])
xp_y=np.array([float(line.split(',')[1]) for line in lines])
f.close()
plt.plot(xp_x,gaussian_filter(xp_y, sigma=40, mode="nearest"), alpha=0.6, linewidth=1, label="3")

f=open("default_init/xpInput_lm0.04_ga0.014CDQN.txt",'r')
lines=f.readlines()
xp_x=np.array([float(line.split(',')[0]) for line in lines])
xp_y=np.array([float(line.split(',')[1]) for line in lines])
f.close()
plt.plot(xp_x,gaussian_filter(xp_y, sigma=40, mode="nearest"), alpha=0.6, linewidth=1, label="4")

f=open("default_init/xpInput_lm0.04_ga0.015CDQN.txt",'r')
lines=f.readlines()
xp_x=np.array([float(line.split(',')[0]) for line in lines])
xp_y=np.array([float(line.split(',')[1]) for line in lines])
f.close()
plt.plot(xp_x,gaussian_filter(xp_y, sigma=40, mode="nearest"), alpha=0.6, linewidth=1, label="5")


#ax.set_xticks([0,250000,500000])
#ax.set_xticklabels([r"$0$",r"$2.5\times10^5$",r"$5\times10^5$"])
#ax.set_yticks(yticks)
#ax.set_yticklabels(["{:.3f}".format(f) for f in yticks])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

ax.set_ylim(bottom=0.)#top=0.4,
#ax.set_xlim(left=0.,right=549444+10000)
#plt.legend(loc=1)
plt.title("Convergent DQN")
plt.tight_layout()
plt.subplots_adjust(left=0.17,bottom=0.155,right=0.975,top=0.9)
plt.savefig("default_init/test_CDQN.png")
plt.clf()

