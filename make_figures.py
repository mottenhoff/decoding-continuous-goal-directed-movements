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

    all_paths = [Path('./finished_runs/cv/delta'),
                 Path('./finished_runs/cv/alphabeta'),
                 Path('./finished_runs/cv/bbhg')] 


    results = {path.stem: gr.get_results(path) for path in all_paths}
    # best_results = {condition: gr.get_best_scores(result) for condition, result in results.items()}
    # best_paths = [ppt['paths'] for ppt in best_results['delta'][0].values()]

    for condition, result in results.items():
        plot_decoding_scores.plot_overview(result, condition)

    # Max score over states,
    # mean performance per kinematic per band
    plot_overview_over_bands.plot(results, all_paths)

    plot_dataset_metrics.plot_average_time_to_target(all_paths[0])
    plot_dataset_metrics.plot_average_trajectory(all_paths[0])
    
    # gaps_vs_performance.plot_relationship(results)  # currently doesnt work


    # TODO:
    # Reconstruction
        # plot_reconstruction_overview.make(path)
        # summarize.main(Path('./results/combined'))
    # Brain correlations
    # Brain combined plot

    # NICE TO HAVE
    # Latent state comparison
        # latent_state_comparisons.main(best_paths)

    