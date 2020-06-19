import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 16}

matplotlib.rc('font', **font)


'''
This part was adapted from:
Asarnow, D., Palovcak, E., Cheng, Y. UCSF pyem v0.5. Zenodo https://doi.org/10.5281/zenodo.3576630 (2019)

Shared on the GNUv3.0
'''

def rot2euler(r):
    """Decompose rotation matrix into Euler angles"""
    # assert(isrotation(r))
    # Shoemake rotation matrix decomposition algorithm with same conventions as Relion.
    epsilon = np.finfo(np.double).eps
    abs_sb = np.sqrt(r[0, 2] ** 2 + r[1, 2] ** 2)
    if abs_sb > 16 * epsilon:
        gamma = np.arctan2(r[1, 2], -r[0, 2])
        alpha = np.arctan2(r[2, 1], r[2, 0])
        if np.abs(np.sin(gamma)) < epsilon:
            sign_sb = np.sign(-r[0, 2]) / np.cos(gamma)
        else:
            sign_sb = np.sign(r[1, 2]) if np.sin(gamma) > 0 else -np.sign(r[1, 2])
        beta = np.arctan2(sign_sb * abs_sb, r[2, 2])
    else:
        if np.sign(r[2, 2]) > 0:
            alpha = 0
            beta = 0
            gamma = np.arctan2(-r[1, 0], r[0, 0])
        else:
            alpha = 0
            beta = np.pi
            gamma = np.arctan2(r[1, 0], -r[0, 0])
    return alpha, beta, gamma

def expmap(e):
    """Convert axis-angle vector into 3D rotation matrix"""
    theta = np.linalg.norm(e)
    if theta < 1e-16:
        return np.identity(3, dtype=e.dtype)
    w = e / theta
    k = np.array([[0, w[2], -w[1]],
                  [-w[2], 0, w[0]],
                  [w[1], -w[0], 0]], dtype=e.dtype)
    r = np.identity(3, dtype=e.dtype) + np.sin(theta) * k + (1 - np.cos(theta)) * np.dot(k, k)
    return r

'''End of adapted code'''


cryosparc_cs_particles = '/path/to/the/file.cs'

data_ = np.load(cryosparc_cs_particles)

poses = data_['alignments3D/pose']
print(poses)

poses_angles = []
for pose in poses:
    poses_angles.append(np.rad2deg(rot2euler(expmap(pose))))

poses_angles = np.stack(poses_angles)


plt.figure(figsize=(12,6))

plt.hist2d(poses_angles[:,0], poses_angles[:,1], cmap='jet', bins=50, norm=LogNorm())
clb = plt.colorbar()
clb.set_label('Number of particles')
plt.xlabel('Azimuth (°)')
plt.ylabel('Elevation (°)')


plt.savefig("cryosparc_angles.svg", dpi=300)
plt.show()