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
from figures import plot_task_correlations

from libs import utils
c = utils.load_yaml('./config.yml')


if __name__=='__main__':


    all_paths = [
        Path(r'finished_runs\delta_cer'),
        Path(r'finished_runs/alphabeta_cer'),
        Path(r'finished_runs\bbhg_cer'),
        Path(r'finished_runs\delta_lap'),
        Path(r'finished_runs/alphabeta_lap'),
        Path(r'finished_runs\bbhg_lap'),
        Path(r'finished_runs\delta_cer_tv'),
        Path(r'finished_runs/alphabeta_cer_tv'),
        Path(r'finished_runs\bbhg_cer_tv')
        ] 


    # results = {path.stem: gr.get_results(path) for path in all_paths}

    # Individual decoding scores per kinematic
    # for condition, result in results.items():
    #     plot_decoding_scores.plot_overview(result, condition)

    # Aggregated decoding performance per kinematic
    # for opt in [
    #     'cer', 
    #     # 'lap', 
    #     # 'cer_tv'
    #     ]:
        
    #     run_results  = {key: value for key, value in results.items() 
    #                     if key in [f'delta_{opt}', 
    #                                f'alphabeta_{opt}',
    #                                f'bbhg_{opt}']}

    #     plot_overview_over_bands.plot(run_results, name=opt)

    # plot_task_correlations.main()

    # plot_dataset_metrics.plot_average_time_to_target(all_paths[0])  # Only first condition because behavior is the same.
    plot_dataset_metrics.plot_speed_curve(all_paths[0])  # Only first condition because behavior is the same.
    plot_dataset_metrics.plot_average_trajectory(all_paths[0])  ## Throws error
    print()
    
    # TODO: Reconstruction plot as example of performance.





    # gaps_vs_performance.plot_relationship(results)  # currently doesnt work


    # Reconstruction
    # plot_reconstruction_overview.make(path)
        # summarize.main(Path('./results/combined'))
    # Brain correlations
    # Brain combined plot

    # NICE TO HAVE
    # Latent state comparison
        # latent_state_comparisons.main(best_paths)

    