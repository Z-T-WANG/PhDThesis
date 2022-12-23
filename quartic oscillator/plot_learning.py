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


figsize = (4,3.2) #s(8,6.4)# 



plt.figure(figsize=figsize)
ax=plt.gca()
plt.xlabel(r"simulation time ($1/\omega_c$)") # ($T$)
plt.ylabel(r"energy ($\hbar\omega_c$)") # $E$

f=open("xpInput_lm0.04_ga0.011.txt",'r')
lines=f.readlines()
xp_x=np.array([float(line.split(',')[0]) for line in lines])
xp_y=np.array([float(line.split(',')[1]) for line in lines])
f.close()
plt.plot(xp_x*2,gaussian_filter(xp_y, sigma=40, mode="nearest"), alpha=0.6, linewidth=1, label="1")
f=open("xpInput_lm0.04_ga0.012.txt",'r')
lines=f.readlines()
xp_x=np.array([float(line.split(',')[0]) for line in lines])
xp_y=np.array([float(line.split(',')[1]) for line in lines])
f.close()
plt.plot(xp_x*2,gaussian_filter(xp_y, sigma=40, mode="nearest"), alpha=0.6,linewidth=1, label="2")

f=open("xpInput_lm0.04_ga0.013.txt",'r')
lines=f.readlines()
xp_x=np.array([float(line.split(',')[0]) for line in lines])
xp_y=np.array([float(line.split(',')[1]) for line in lines])
f.close()
plt.plot(xp_x*2,gaussian_filter(xp_y, sigma=40, mode="nearest"), alpha=0.6,linewidth=1, label="3")

f=open("xpInput_lm0.04_ga0.014.txt",'r')
lines=f.readlines()
xp_x=np.array([float(line.split(',')[0]) for line in lines])
xp_y=np.array([float(line.split(',')[1]) for line in lines])
f.close()
plt.plot(xp_x*2,gaussian_filter(xp_y, sigma=40, mode="nearest"), alpha=0.6,linewidth=1, label="4")

f=open("xpInput_lm0.04_ga0.015.txt",'r')
lines=f.readlines()
xp_x=np.array([float(line.split(',')[0]) for line in lines])
xp_y=np.array([float(line.split(',')[1]) for line in lines])
f.close()
plt.plot(xp_x*2,gaussian_filter(xp_y, sigma=40, mode="nearest"), alpha=0.6,linewidth=1, label="5")


#ax.set_xticks([0,250000,500000])
#ax.set_xticklabels([r"$0$",r"$2.5\times10^5$",r"$5\times10^5$"])
#ax.set_yticks(yticks)
#ax.set_yticklabels(["{:.3f}".format(f) for f in yticks])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

ax.set_ylim(bottom=0.)#top=0.4,
#ax.set_xlim(left=0.,right=549444+10000)
#plt.legend(loc=1)
ax.ticklabel_format(axis="x", style="scientific", scilimits = (0,0))
plt.title("DQN")
plt.tight_layout()
plt.subplots_adjust(left=0.17,bottom=0.155,right=0.975,top=0.9)
plt.savefig("test_DQN.pdf")
plt.close()





plt.figure(figsize=figsize)
ax=plt.gca()
plt.xlabel(r"simulation time ($1/\omega_c$)") # ($T$)
plt.ylabel(r"energy ($\hbar\omega_c$)") # $E$

f=open("xpInput_lm0.04_ga0.011CDQN.txt",'r')
lines=f.readlines()
xp_x=np.array([float(line.split(',')[0]) for line in lines])
xp_y=np.array([float(line.split(',')[1]) for line in lines])
f.close()
plt.plot(xp_x*2,gaussian_filter(xp_y, sigma=40, mode="nearest"), alpha=0.6, linewidth=1, label="1")
f=open("xpInput_lm0.04_ga0.012CDQN.txt",'r')
lines=f.readlines()
xp_x=np.array([float(line.split(',')[0]) for line in lines])
xp_y=np.array([float(line.split(',')[1]) for line in lines])
f.close()
plt.plot(xp_x*2,gaussian_filter(xp_y, sigma=40, mode="nearest"), alpha=0.6, linewidth=1, label="2")

f=open("xpInput_lm0.04_ga0.013CDQN.txt",'r')
lines=f.readlines()
xp_x=np.array([float(line.split(',')[0]) for line in lines])
xp_y=np.array([float(line.split(',')[1]) for line in lines])
f.close()
plt.plot(xp_x*2,gaussian_filter(xp_y, sigma=40, mode="nearest"), alpha=0.6, linewidth=1, label="3")

f=open("xpInput_lm0.04_ga0.014CDQN.txt",'r')
lines=f.readlines()
xp_x=np.array([float(line.split(',')[0]) for line in lines])
xp_y=np.array([float(line.split(',')[1]) for line in lines])
f.close()
plt.plot(xp_x*2,gaussian_filter(xp_y, sigma=40, mode="nearest"), alpha=0.6, linewidth=1, label="4")

f=open("xpInput_lm0.04_ga0.015CDQN.txt",'r')
lines=f.readlines()
xp_x=np.array([float(line.split(',')[0]) for line in lines])
xp_y=np.array([float(line.split(',')[1]) for line in lines])
f.close()
plt.plot(xp_x*2,gaussian_filter(xp_y, sigma=40, mode="nearest"), alpha=0.6, linewidth=1, label="5")


#ax.set_xticks([0,250000,500000])
#ax.set_xticklabels([r"$0$",r"$2.5\times10^5$",r"$5\times10^5$"])
#ax.set_yticks(yticks)
#ax.set_yticklabels(["{:.3f}".format(f) for f in yticks])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

ax.set_ylim(bottom=0.)#top=0.4,
ax.ticklabel_format(axis="x", style="scientific", scilimits = (0,0))
#plt.legend(loc=1)
plt.title("Convergent DQN")
plt.tight_layout()
plt.subplots_adjust(left=0.17,bottom=0.155,right=0.975,top=0.9)
plt.savefig("test_CDQN.pdf")
plt.close()



figsize = (15,3.5)
fig, axes = plt.subplots(1,5,figsize=figsize)
#plt.ticklabel_format(axis="x", style="sci")
sigma = 4
for i, ax in enumerate(axes):
    seed = i + 1
    ax.set_title("seed {}".format(seed))
    f=open("xpInput_lm0.04_ga0.01{}.txt".format(seed),'r')
    lines=f.readlines()
    xp_x=np.array([float(line.split(',')[0]) for line in lines])
    xp_y=np.array([float(line.split(',')[1]) for line in lines])
    f.close()
    ax.ticklabel_format(axis="x", style="scientific", scilimits = (0,0))
    xp_y = gaussian_filter(xp_y, sigma=sigma, mode="nearest")
    ax.plot(xp_x*2,xp_y, alpha=0.8, linewidth=1, label="DQN")
    f=open("xpInput_lm0.04_ga0.01{}CDQN.txt".format(seed),'r')
    lines=f.readlines()
    xp_x=np.array([float(line.split(',')[0]) for line in lines])
    xp_y=np.array([float(line.split(',')[1]) for line in lines])
    f.close()
    xp_y = gaussian_filter(xp_y, sigma=sigma, mode="nearest")
    ax.plot(xp_x*2,xp_y, alpha=0.8, linewidth=1, label="C-DQN")
    ax.set_ylim(bottom=0., top=12.)
    if seed == 1:
        ax.set_xlabel(r"simulation time ($1/\omega_c$)")
        ax.set_ylabel(r"energy ($\hbar\omega_c$)")
        ax.legend(loc="upper right")
        ax.xaxis.set_label_coords(0.43,-0.113)

#ax.set_xticks([0,250000,500000])
#ax.set_xticklabels([r"$0$",r"$2.5\times10^5$",r"$5\times10^5$"])
#ax.set_yticks(yticks)
#ax.set_yticklabels(["{:.3f}".format(f) for f in yticks])

#ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

plt.tight_layout()
plt.subplots_adjust(left=0.04,right=0.99,top=0.92,bottom=0.15) #left=0.17,bottom=0.155,right=0.975,top=0.9
plt.subplots_adjust(wspace = 0.2) 
plt.savefig("DQN_CDQN_results_seeds_sigma{}.pdf".format(sigma))
plt.close()


figsize = (15,3.5)
fig, axes = plt.subplots(1,5,figsize=figsize)
#plt.ticklabel_format(axis="x", style="sci")
sigma = 20
for i, ax in enumerate(axes):
    seed = i + 1
    ax.set_title("seed {}".format(seed))
    f=open("xpInput_lm0.04_ga0.01{}.txt".format(seed),'r')
    lines=f.readlines()
    xp_x=np.array([float(line.split(',')[0]) for line in lines])
    xp_y=np.array([float(line.split(',')[1]) for line in lines])
    xp_y=((xp_y==12.).astype(int)).astype(float)
    f.close()
    ax.ticklabel_format(axis="x", style="scientific", scilimits = (0,0))
    xp_y = gaussian_filter(xp_y, sigma=sigma, mode="nearest")
    ax.plot(xp_x*2,xp_y, alpha=0.8, linewidth=1, label="DQN")
    f=open("xpInput_lm0.04_ga0.01{}CDQN.txt".format(seed),'r')
    lines=f.readlines()
    xp_x=np.array([float(line.split(',')[0]) for line in lines])
    xp_y=np.array([float(line.split(',')[1]) for line in lines])
    xp_y=((xp_y==12.).astype(int)).astype(float)
    f.close()
    xp_y = gaussian_filter(xp_y, sigma=sigma, mode="nearest")
    ax.plot(xp_x*2,xp_y, alpha=0.8, linewidth=1, label="C-DQN")
    ax.set_ylim(top=1.)
    if seed == 1:
        ax.set_xlabel(r"simulation time ($1/\omega_c$)")
        ax.set_ylabel(r"failure rate")
        ax.legend(loc="upper right")
        ax.xaxis.set_label_coords(0.43,-0.113)

#ax.set_xticks([0,250000,500000])
#ax.set_xticklabels([r"$0$",r"$2.5\times10^5$",r"$5\times10^5$"])
#ax.set_yticks(yticks)
#ax.set_yticklabels(["{:.3f}".format(f) for f in yticks])

#ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

plt.tight_layout()
plt.subplots_adjust(left=0.04,right=0.99,top=0.92,bottom=0.15) #left=0.17,bottom=0.155,right=0.975,top=0.9
plt.subplots_adjust(wspace = 0.2) 
plt.savefig("DQN_CDQN_failure_seeds_sigma{}.pdf".format(sigma))
plt.close()




