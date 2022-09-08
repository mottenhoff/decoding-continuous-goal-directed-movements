import matplotlib.pyplot as plt
import numpy as np

def plot_trajectory(xyz, events):

    xyz_cursor_reset =     xyz[events.cursor_reset_start[:, 0].astype(int)-1, :]
    xyz_cursor_reset_end = xyz[events.cursor_reset_end[:, 0].astype(int), :]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    ax.scatter(events.target_new[:, 1], events.target_new[:, 2], events.target_new[:, 3], s=40, c='r', label='Targets')
    ax.scatter(xyz_cursor_reset[:, 0], xyz_cursor_reset[:, 1], xyz_cursor_reset[:, 2], c='g', label='Hand off screen')
    ax.scatter(xyz_cursor_reset_end[:, 0], xyz_cursor_reset_end[:, 1], xyz_cursor_reset_end[:, 2], c='y', label='Hand back on screen')
    
    ax.set_xlabel('x [left right]')
    ax.set_ylabel('y [up down]')
    ax.set_zlabel('z [depth]')

    ax.legend()
    plt.show()

def check_distance(xyz, targets):

    xyz = xyz[targets[:, 0].astype(int), :]

    d = np.sqrt(((xyz-targets[:, 1:])**2).sum(axis=1))

    plt.figure()
    plt.hist(d, bins=20)
    plt.title('mm to target')
    plt.xlabel('mm')
    plt.ylabel('count')
