from figures import figure_3d_correlation_and_reconstruction
from figures import figure_1d_score_overview

from figures import plot_reconstruction_overview

from libs import utils
c = utils.load_yaml('./config.yml')

def plot_speed(path):

    figure_1d_score_overview.make(path)

def make_overview(path):
    if c.complete_model:
        pass
        return
    
    if c.pos:
        pass
    
    if c.vel:
        pass

    if c.speed:
        figure_speed_overview.make(path)


def make(path):

    if c.speed:
        plot_speed(path)

    if c.pos or c.vel:
        figure_3d_correlation_and_reconstruction.make(path)


