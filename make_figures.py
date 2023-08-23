from pathlib import Path

# from figures import figure_3d_correlation_and_reconstruction
# from figures import figure_1d_score_overview
# from figures import figure_6d_score_overview

from figures import plot_reconstruction_overview
# from figures.all_figures import make_overview

# from figures import summarize, 
from figures import plot_decoding_scores
from figures import plot_overview_over_bands
from figures import plot_significant_channels
from figures import gaps_vs_performance
from figures import latent_state_comparisons
from figures import plot_dataset_metrics

from libs import utils
c = utils.load_yaml('./config.yml')



if __name__=='__main__':

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



    # path = Path('./results/ab')
    # path = Path('./results/delta')
    # path = Path('./finished_runs/window/bbhg')

    all_paths = [Path('./finished_runs/window/delta'),
                 Path('./finished_runs/window/alphabeta'),
                 Path('./finished_runs/window/bbhg')] 

    results = [plot_decoding_scores.plot_overview(path) 
                    for path in all_paths]

    plot_overview_over_bands.plot(results, all_paths)
    plot_dataset_metrics(all_paths[0])




    # best_paths, scores = plot_decoding_scores.plot_overview(path)

    # gaps_vs_performance.plot_relationship(best_paths)
    # latent_state_comparisons.main(best_paths)
    plot_dataset_metrics.all(best_paths)

    for path in best_paths:
        # plot_reconstruction_overview.make(path)
        pass


    # plot_significant_channels.plot(best_paths, scores)
    # summarize.main(Path('./results/combined'))
    print('')



    # [] Plot speed from trial to trial.