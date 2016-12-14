from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import files

if __name__ == "__main__":
    load = files.hdf5Load("final.hdf5")

    t = load[0]
    R = load[1:]

    dim = 3 
    R = R.reshape((len(R)//dim,dim,R.shape[1]))
    r = R[::2]

    #r = np.insert(r,2,0,axis=1).T
    r = r.T
    
    v = R[1::dim]
    dpi = 192
    fig = plt.figure(figsize=(38.40,21.60))
    ax = fig.add_subplot(111, projection='3d')
    scat = ax.scatter([],[],[],s=12,c="black")
    bb = 1
    ax.set_xlim3d(-bb, bb)
    ax.set_ylim3d(-bb, bb)
    ax.set_zlim3d(-bb, bb)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.azim = 0
    ax.elev= 0

    def animate(i):
        print i
        if ax.elev<45:
            ax.elev += 0.2
        ax.azim += 0.2
        scat._offsets3d=r[i]

    myframes = np.arange(len(t))[::4]
    myframes = myframes[500:]

    #ani = animation.FuncAnimation(fig,animate,interval=17,frames=myframes) 
    ani = animation.FuncAnimation(fig,animate,frames=myframes) 
    ani.save("animate.mp4",writer="ffmpeg",fps=60,bitrate=60000,codec="libx264",extra_args=["-pix_fmt","yuv420p","-movflags","+faststart","-bf","2","-g","30"])
    #plt.show()
    plt.close()
