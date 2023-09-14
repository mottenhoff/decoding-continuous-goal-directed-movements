from pathlib import Path

# from figures import figure_3d_correlation_and_reconstruction
# from figures import figure_1d_score_overview
# from figures import figure_6d_score_overview

from figures import plot_reconstruction_overview
# from figures.all_figures import make_overview

# from figures import summarize, 
from figures import plot_decoding_scores
from figures import plot_overview_over_bands
from figures import gaps_vs_performance
from figures import latent_state_comparisons
from figures import plot_dataset_metrics
from figures import get_results as gr

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


    
    results = {path.stem: gr.get_results(path) for path in all_paths}
    best_results = {condition: gr.get_best_scores(result) for condition, result in results.items()}
    best_paths = [ppt['paths'] for ppt in best_results['delta'][0].values()]

    for condition, result in best_results.items():
        plot_decoding_scores.plot_overview(result[0], condition)

    # Max score over states,
    # mean performance per kinematic per band
    plot_overview_over_bands.plot(results, all_paths)

    plot_dataset_metrics.plot_average_time_to_target(all_paths[0])
    plot_dataset_metrics.plot_average_trajectory(all_paths[0])
    
    gaps_vs_performance.plot_relationship(best_paths)


    # TODO:
    # Reconstruction
        # plot_reconstruction_overview.make(path)
        # summarize.main(Path('./results/combined'))
    # Brain correlations
    # Brain combined plot

    # NICE TO HAVE
    # Latent state comparison
        # latent_state_comparisons.main(best_paths)

    