import matplotlib.pyplot as plt
import numpy as np

# Create figure and 3D axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define the cursor path as a semi-circular arc
theta = np.linspace(0, np.pi, 100)
r = 5
x = r * np.cos(theta)
y = -r * np.sin(theta) + 5
z = np.linspace(0, 4, 100)
# z = np.zeros_like(theta)

# Plot the cursor path
ax.plot(x, y, z, 'b->', label='Cursor Path', color='b')
# ax.arrow(x, y, z, 'b', label='Cursor Path', color='b', head_width='0.05')

# Define the target point
target_x = 0
target_y = r
target_z = 5

# Plot the target point
ax.scatter(target_x, target_y, target_z, color='r', label='Target')

# Plot the target vectors (dotted lines from cursor path to the target point)
for t in range(0, 100, 20):
    ax.plot([x[t], target_x], [y[t], target_y], [z[t], target_z], 'k--', alpha=0.5, label='Target vector' if t==0 else '')

# Labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set the aspect of the plot to be equal
ax.set_box_aspect([1, 1, 1])  # aspect ratio is 1:1:1

# Set plot limits
ax.set_xlim([-5, 5])
ax.set_ylim([0, 5])
ax.set_zlim([0, 5])

# Add legend
ax.legend()

# Show plot
plt.show()
