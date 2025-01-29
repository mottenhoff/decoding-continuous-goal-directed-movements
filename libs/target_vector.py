import matplotlib.pyplot as plt

def plot_vectors(xyz, trials, target_vector):
	# NOTE: trials[:, 0] is trial number

	fig, ax = plt.subplots(nrows=1, ncols=2)
	ax[0].scatter(xyz[:, 0], xyz[:, 1], color='black')
	ax[0].scatter(trials[:, 1], trials[:, 2], color='red')

	for i in range(len(xyz)):
			ax[0].arrow(xyz[i, 0], xyz[i, 1], target_vector[i, 0], target_vector[i, 1], color='blue', head_width=0.2)

	ax[0].set_xlabel('x')
	ax[0].set_ylabel('y')

	fig.show()

def get_target_vector(trials, xyz):

	target_vector = trials[:, 1:] - xyz  # Triggers if first value = nan

	if False:
		plot_vectors(xyz, trials, target_vector)

	return target_vector