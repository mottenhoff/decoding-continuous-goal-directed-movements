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

def velocity(xyz, ts=None):
    
    if ts is None:
        # Assumes equal timesteps
        return np.diff(xyz, axis=0)
    
    return (np.diff(xyz, axis=0).T / np.diff(ts)).T

def vector_length(vec_2d):
    # Length of vector
    # = np.sqrt((xyz**2).sum(axis=1))
    return np.linalg.norm(vec_2d, axis=1)

if __name__=='__main__':
    from pathlib import Path
    from read_xdf import read_xdf
    
    path = Path(r'./data/kh040/bubbles_2.xdf')
    data = read_xdf(path)[0]['LeapLSL']
    xyz = data['data'][:, 7:10]
    ts =  data['ts']

    v = velocity(xyz)
    s = vector_length(v)
    p = cart_to_pol(xyz)