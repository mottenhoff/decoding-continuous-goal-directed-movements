import numpy as np

def get_target_vector(xyz, trials):

    target_vec = trials - xyz
     

    return target_vec

def cart_to_pol(xyz):
    # Cartesian to polar coordinates
    rho = vector_length(xyz)                            # Length of vector
    phi = np.arctan(xyz[:, 1]/xyz[:, 0])                # Angle x and y
    theta = np.arccos(vector_length(xyz[:, :2]) / rho)  # Angle xy-plane and z
    return np.hstack((rho, phi, theta))

def differentiate(xyz, ts):
    # prepend a linaer extrapolation to assume contant diff in the first 2 samples
    dx = np.diff(xyz, axis=0, prepend=xyz[:1, :] - (xyz[1,:] - xyz[0, :]))
    dt = np.diff(ts,  axis=0, prepend=ts[0] - (ts[1] - ts[0]))[:, np.newaxis]

    return dx/dt

def vector_length(vec_2d):
    # Length of vector
    # = np.sqrt((xyz**2).sum(axis=1))
    return np.linalg.norm(vec_2d, axis=1, keepdims=True)

def get_all(xyz, ts):

    velocity = differentiate(xyz, ts)
    acceleration = differentiate(velocity, ts)

    distance = vector_length(xyz)
    speed = vector_length(velocity)
    force = vector_length(acceleration)
    
    return np.hstack([
                xyz,
                velocity,
                acceleration,
                distance,
                speed,
                force])