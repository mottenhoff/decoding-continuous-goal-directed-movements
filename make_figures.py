from pathlib import Path

import sys
import matplotlib.pyplot as plt

sys.path.append(r'/home/maarten/main/resources/code/')
# from figures import figure_3d_correlation_and_reconstruction
# from figures import figure_1d_score_overview
# from figures import figure_6d_score_overview

# from figures import plot_reconstruction_overview
# from figures.all_figures import make_overview

# from figures import summarize, 
from figures import plot_decoding_scores
from figures import plot_overview_over_bands
from figures import gaps_vs_performance
from figures import latent_state_comparisons
from figures import plot_dataset_metrics
from figures import get_results as gr
from figures import plot_task_correlations
from figures import plot_dpad_vs_psid

# from figures import plot_3d_brains_significance

# from libs import utils
# c = utils.load_yaml('./config.yml')


if __name__=='__main__':
    path_data = Path(r'../data')
    path_results_dpad = Path(r'./results_dpad')
    path_results_psid = Path(r'./results_psid')
    # path_results_psid = Path(r'./finished_runs')

    path_results = path_results_dpad

    savepath = Path('figures_'+path_results.name.split('_')[-1])
    savepath.mkdir(parents=True, exist_ok=True)

    all_paths = [
        path_results/'raw_cer',
        path_results/'delta_cer',
        path_results/'bbhg_cer',
        path_results/'alphabeta_cer',
        # path_results/'delta_lap',
        # path_results/'alphabeta_lap',
        # path_results/'bbhg_lap',
        # path_results/'delta_cer_tv',
        # path_results/'alphabeta_cer_tv',
        # path_results/'bbhg_cer_tv'
        ] 

    # plot_3d_brains_significance.main()

    results = {path.stem: gr.get_results(path, path_data) for path in all_paths}

    # # # # Individual decoding scores per kinematic
    for condition, result in results.items():
        plot_decoding_scores.plot_overview(result, condition, savepath)

    # # Aggregated decoding performance per kinematic
    for opt in [
        'cer', 
        # 'lap', 
        # 'cer_tv'
        ]:
        
        run_results  = {key: value for key, value in results.items() 
                        if key in [f'delta_{opt}', 
                                   f'alphabeta_{opt}',
                                   f'bbhg_{opt}']}

        plot_overview_over_bands.plot(run_results, name=opt, savepath=savepath)

    # plot_task_correlations.main(path_results, savepath)

    # plot_dataset_metrics.plot_average_time_to_target(all_paths[0], savepath)  # Only first condition because behavior is the same.
    # # plot_dataset_metrics.plot_speed_curve(all_paths[0], savepath)  # Only first condition because behavior is the same.
    # plot_dataset_metrics.plot_average_trajectory(all_paths[0], savepath)  ## Throws error

    psid_paths = [
        path_results_psid/'raw_cer',
        path_results_psid/'delta_cer',
        path_results_psid/'alphabeta_cer',
        path_results_psid/'bbhg_cer',
        # path_results/'delta_lap',
        # path_results/'alphabeta_lap',
        # path_results/'bbhg_lap',
        # path_results/'delta_cer_tv',
        # path_results/'alphabeta_cer_tv',
        # path_results/'bbhg_cer_tv'
        ] 


    results_dpad = results
    results_psid = {path.stem: gr.get_results(path, path_data) for path in psid_paths}

    fig, axs= plt.subplots(nrows=1, ncols=4, figsize=(24, 8))
    for i, condition in enumerate(['raw_cer', 'delta_cer', 'alphabeta_cer', 'bbhg_cer']):
        axs[i] = plot_dpad_vs_psid.plot(results_dpad[condition], results_psid[condition], ax=axs[i])
        axs[i].set_title(condition.split('_')[0].capitalize(), fontsize='xx-large')
        axs[i].set_aspect('equal', 'box')

    axs[-1].legend(bbox_to_anchor=(1.1, 1))
    plt.show(block=True)