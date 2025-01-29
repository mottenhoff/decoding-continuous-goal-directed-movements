from random import randint
from math import sin, cos, inf
import numpy as np
import matplotlib.pyplot as plt

def pythagoras(a, b, c):
    return np.sqrt(a**2+b**2+c**2)

def generate_point(c, l, a1, a2):
    z = l*sin(a1)
    z_p = l*cos(a1)
    x = z_p*cos(a2)
    y = z_p*sin(a2)

    return np.array((c[0]+x, c[1]+y, c[2]+z)).ravel()

def is_in_boundary(p, b):
    return True if p >= b[0] and p <= b[1] else False

def generate_trials(n, box_x, box_y, box_z, margin=0.03):

    dx = box_x[1]-box_x[0]
    dy = box_y[1]-box_y[0]
    dz = box_z[1]-box_z[0]

    bx = [box_x[0]+dx*margin, box_x[1]-dx*margin]
    by = [box_y[0]+dy*margin, box_y[1]-dy*margin]
    bz = [box_z[0]+dz*margin, box_z[1]-dz*margin]

    print('Generating for box:', bx, by, bz)

    middle = (dx//2, dy//2, dz//2)
    max_possible_length = pythagoras(dx, dy, dz)
    max_length = 0.95*max_possible_length
    min_length = 0.05*max_possible_length
    print('Length [min, max]', min_length, max_length)

    # Generate random direction
    lengths = np.random.uniform(min_length, max_length, n)

    targets = []
    used_lengths = []
    attempts = 0
    max_attempts = 100
    prev_target = middle
    xyz = [inf, inf, inf]
    for l in lengths:
        while True:
            a1 = randint(0, 360)
            a2 = randint(0, 360)
            xyz = generate_point(prev_target, l, a1, a2)

            if all([is_in_boundary(xyz[0], bx),
                    is_in_boundary(xyz[1], by),
                    is_in_boundary(xyz[2], bz)]):
                targets += [xyz]
                used_lengths += [float(l)]
                prev_target = xyz
                attempts = 0
                print(len(used_lengths))
                break
            
            attempts += 1
            if attempts == max_attempts:
                print(f'Failed {max_attempts} times!')
                l = np.random.uniform(min_length, max_length, 1)
                attempts = 0

    targets = np.vstack(targets)
    used_lengths = np.array(lengths)
    lengths_calc = pythagoras(targets[:, 0], targets[:, 1], targets[:, 2])

    # print(used_lengths, lengths_calc)

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(targets[:,0], targets[:,1], targets[:,2], c=range(targets.shape[0]), cmap='inferno')
    plt.figure()
    plt.hist(pythagoras(*np.diff(targets, axis=0).T), bins=100)

    fig, ax = plt.subplots()
    plt.hist(used_lengths, bins=100)
    ax.axvline(min_length, c='r')
    ax.axvline(max_length, c='r')

    plt.show()

    print(targets.min(axis=0), targets.max(axis=0))
    # print(xyz, 'length=', sqrt(xyz[0]**2+xyz[1]**2+xyz[2]**2))

    # SAVE
    # with open('trials.npy', 'wb') as f:
    #     np.save(f, targets)

    return targets

    
if __name__=='__main__':
    generate_trials(1000, (0, 1600), (0, 900), (10, 110), 0.1)
