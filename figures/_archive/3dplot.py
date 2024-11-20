def 3dplot(xyz):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')

    # ax.set_title('Trajectory [Highest total correlation]')
    
    z = xyz[~np.isnan(xyz[:, :3]).all(axis=1), :]
    # z_hat = zh.squeeze()  # Might change when multiple options used
    # ax.plot(z_hat[:, 0], z_hat[:, 1], z_hat[:, 2], label='Predicted')
    ax.plot(z[:, 0], z[:, 1], z[:, 2], label='True')

    ax.set_xlabel('X [left - right]', fontsize='x-large')
    ax.set_ylabel('Y [up - down]', fontsize='x-large')
    ax.set_zlabel('Z [front - back]', fontsize='x-large')
    ax.set_title('Hand Trajectory', fontsize='xx-large')
    # ax.legend()

    ax.view_init(azim=45, elev=45)

    fig.savefig('3d.png')