import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy, math
import colorsys, cv2

def complex_to_rgb(z): # this uses the CIELCh space, fixing C and rotating the hue
    #H = np.angle(z)/(2.*math.pi)+0.5 + 1./3 # + is green and - is magenta, i is sky blue and -i is orange
    #S = np.full_like(H, 1.)
    z = z[4:-4, 4:-4]
    norm = np.abs(z)
    L = (1.- 1./(1+norm**2)) # * 100.
    L = np.sqrt(L) ### to smoothen lightness
    norm[norm==0.]=1.
    z_ = z/norm 
    z_[np.isnan(z_)] = 0.
    C = 127.
    a = z_.real * C
    b = z_.imag * C
    #HLS = np.stack([H, L, S], axis = -1))
    Lab = np.stack([np.full_like(a, 50.),a,b], axis=-1).astype(np.float32)
    RGB = cv2.cvtColor(Lab, cv2.COLOR_Lab2RGB)
    
    # setting brightness vs. probability density:
    HLS = cv2.cvtColor(RGB, cv2.COLOR_RGB2HLS)
    h,l,s = cv2.split(HLS)
    s = np.clip(s, 0., 1.)
    h = np.clip(h, 0., 360.)
    l = L/2. # invert brightness
    RGB = cv2.cvtColor(cv2.merge([h,l.astype(np.float32),s]), cv2.COLOR_HLS2RGB)
    
    return RGB

def save_plot_close(file_name, task_name, Q_z, gamma):
    plt.tight_layout(rect=[-0.13, 0, 0.93, 1.],h_pad=1.5)
    if not os.path.isdir("{}/{:.2f}_{}".format(task_name, gamma, Q_z)): os.makedirs("{}/{:.2f}_{}".format(task_name, gamma, Q_z))
    plt.savefig("{}/{:.2f}_{}/{}.png".format(task_name, gamma, Q_z, file_name), bbox_inches='tight')
    plt.close()

def plot3d_and_2d(state):
    fig = plt.figure(figsize=plt.figaspect(2))
    ax1 = fig.add_subplot(2,1,1, projection='3d', proj_type = 'ortho')
    X, Y = np.meshgrid(x_array, x_array)
    ax1.plot_wireframe(X, Y, np.abs(state), rstride=5, cstride=5)
    ax2 = fig.add_subplot(2,1,2)
    ax2.imshow(plot.complex_to_rgb(state/plot_downscale), extent = [x_array[0]-grid_size/2., x_array[-1]+grid_size/2., x_array[0]-grid_size/2., x_array[-1]+grid_size/2.], origin='lower')
    ax2.yaxis.set_label_position("right")
    return ax1, ax2

# if ax2Image is None:
#     ax2Image = imshow(complex_to_rgb(state/plot_downscale), extent = [x_array[0]-grid_size/2., x_array[-1]+grid_size/2., x_array[0]-grid_size/2., x_array[-1]+grid_size/2.], origin='lower')
# else:
#     ax2Image.set(data = complex_to_rgb(state/plot_downscale))
# ax1.clear()
# ax1.plot_wireframe(X, Y, np.abs(state), rstride=5, cstride=5)
##### fig.canvas.draw(); fig.canvas.flush_events()









