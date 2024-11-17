import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as scio

parent = [16, 15, 1, 2, 3, 1, 5, 6, 14, 8, 9, 14, 11, 12, 14, 14, 1]
# data = scio.loadmat('../checkpoint/inference_data.mat')
# data = scio.loadmat('../checkpoint/Eval/inference_data.mat')
# data = scio.loadmat('/home/final/HD/Code/ERPoseFormer_v2/checkpoint/Eval/mixcoco_31.04_81_STMO/inference_data.mat')
data = scio.loadmat('/home/final/HD/Code/ERPoseFormer_v2/checkpoint/3DHP/visualize_mat_file/inference_data_f81_30.8.mat')

joints_right=[2, 3, 4, 8, 9, 10]


#data_3d = data["TS1"][:,:,:,100]
#data_3d = data["TS4"][:,:,:,80]
data_3d = data["TS6"][:,:,:,10]
data_3d = np.squeeze(data_3d,axis = 2)
data_3d=np.transpose(data_3d,(1,0))

data_3d = data_3d - data_3d[14:15]

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

xy_radius=1000
radius=1500
ax.view_init(elev=15., azim=-70)
ax.set_xlim3d([-xy_radius / 2, xy_radius / 2])
ax.set_zlim3d([-radius / 2, radius / 2])
ax.set_ylim3d([-xy_radius / 2, xy_radius / 2])
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
ax.dist = 8
ax.set_title("Ours")  # , pad=35
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.get_zaxis().set_visible(False)
#ax.set_axis_off()


for i in range(17):
    col = 'yellowgreen' if i in joints_right else 'midnightblue'
    ax.plot([data_3d[i, 0], data_3d[parent[i], 0]], [data_3d[i, 2], data_3d[parent[i], 2]], [-data_3d[i, 1], -data_3d[parent[i], 1]], c=col )
    #ax.annotate(s=str(i), x=data_2d[i,0], y=data_2d[i,1]-10,color='white', fontsize='3')

#plt.show()
save_3d_folder_path = "/home/final/HD/Data/3DHPE/Visualization/3DHP/3D/"
save_3d_file = "3dhp_test_3d.png"
# plt.savefig(save_3d_folder_path+save_3d_file, bbox_inches="tight", pad_inches=0.0, dpi=300)
plt.savefig("/home/final/HD/Data/3DHPE/Visualization/3DHP/3D/3dhp_test_3d.png", bbox_inches="tight", pad_inches=0.0, dpi=300)
# plt.savefig("./3dhp_test_3d.png", bbox_inches="tight", pad_inches=0.0, dpi=300)
plt.close()