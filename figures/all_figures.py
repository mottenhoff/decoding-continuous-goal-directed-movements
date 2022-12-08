from figures import figure_3d_correlation_and_reconstruction
from figures import figure_1d_score_overview
from figures import figure_6d_score_overview

from figures import figure_speed_overview

from libs import utils
c = utils.load_yaml('./config.yml')

def plot_speed(path):
    # TODO: Session path
    if c.complete_model:
        figure_6d_score_overview.make(path)
    else:
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


if __name__=='__main__':
    from pathlib import Path
    path = Path('/home/coder/project/results/20221207_0215')
    make_overview(path)