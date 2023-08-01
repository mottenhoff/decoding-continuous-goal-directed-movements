from figures import figure_3d_correlation_and_reconstruction
from figures import figure_1d_score_overview
from figures import figure_6d_score_overview

from figures import figure_speed_overview
from figures.all_figures import make_overview

from figures import summarize, plot_decoding_scores


from libs import utils
c = utils.load_yaml('./config.yml')



if __name__=='__main__':
    from pathlib import Path
    # path = Path('/home/coder/project/results/20221207_0215')
    # path = Path('/home/coder/project/results/20221209_0257')
    # path = Path('finished_runs/delta_all_car/20230526_1752/')  # 5_5_10
    # path = Path('results/20230714_1301')
    # path = Path('results/20230616_0733')
    # make_overview(path)

    # summarize.main(Path('./results/20230717_0927'))  # Path('./finished_runs/beta_all/')
    # summarize.main(Path('./finished_runs/delta_all_car/'))  # Path('./finished_runs/beta_all/')
    # summarize.main(Path('./results'))

    # summarize.main(Path('./results/20230728_1700'))

    path = Path('./results/new_method')
    plot_decoding_scores.plot_overview(path)

    # summarize.main(Path('./results/combined'))