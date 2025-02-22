import numpy as np

def get_target_vector(xyz, trials):
    return trials - xyz

def cart_to_pol(xyz):
    # Cartesian to polar coordinates
    rho = vector_length(xyz)                            # Length of vector
    phi = np.arctan(xyz[:, 1]/xyz[:, 0])                # Angle x and y
    theta = np.arccos(vector_length(xyz[:, :2]) / rho)  # Angle xy-plane and z
    return np.hstack((rho, phi, theta))

def differentiate(xyz, ts):
    # prepend a linaer extrapolation to assume contant diff in the first 2 samples
    dx = np.diff(xyz, axis=0, prepend=np.zeros((1, 3)))
    dt = np.diff(ts,  axis=0, prepend=0)[:, np.newaxis]

    return dx/dt

def vector_length(vec_2d):
    return np.linalg.norm(vec_2d, axis=1, keepdims=True)

def replace_values_in_new_matrix(base_matrix, indices, values):
    new_matrix = base_matrix.copy()

    if values.shape[1] > 1:
        new_matrix[indices, :] = values
    else:
        new_matrix[indices, :1] = values
        new_matrix = new_matrix[:, :1]

    return new_matrix


def get_all(subset, has_target_vector=False):

    has_value = np.where(~np.isnan(subset.xyz[:, 0]))[0]

    xyz = subset.xyz[has_value, :]
    ts = subset.ts[has_value]

    velocity = differentiate(xyz, ts)
    acceleration = differentiate(velocity, ts)

    if has_target_vector:
        # replace tha value at those indices with 0 in the differentiated vectors
        # otherwise there will be an artificial spike of speed/acceleration because
        # of the switch of target

        # Identify where there is a change of target
        trials = subset.trials
        trial_idc = np.where(~np.isnan(trials[:, 0]))[0]
        new_trials = np.where(np.diff(trials[trial_idc, 0]))[0]

        if any(new_trials):
            # Correct values are those locations with previous value s
            velocity[new_trials+1, :] =     velocity[new_trials, :]
            acceleration[new_trials+1, :] = acceleration[new_trials, :]

            try:
                # One extra because progation from differentiation from speed. (one up and one down)
                acceleration[new_trials+2, :] = acceleration[new_trials, :]
            except IndexError:
                pass

    distance = vector_length(xyz)
    speed = vector_length(velocity)
    force = vector_length(acceleration)

    kinematics = np.hstack([
        velocity,
        acceleration,
        distance, 
        speed,
        force
    ])

    # Map back to original size
    expanded_kinematics = np.full(shape = (subset.xyz.shape[0], kinematics.shape[1]), fill_value=np.nan)
    expanded_kinematics[has_value, :] = kinematics

    return np.hstack([subset.xyz, expanded_kinematics])