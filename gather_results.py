import matplotlib.pyplot as plt
import settings
import os
import csv
import seaborn as sns
import scipy.stats as ss
import numpy as np

np. set_printoptions(suppress=True)
plt.rc('font', family='serif')
plt.rcParams['legend.title_fontsize'] = 16
plt.rcParams['lines.linewidth'] = 2.5

if __name__ == "__main__":
    th_distrib = []
    all_res_dict = {}
    res_matrix = []

    for dataset_id, dataset_name in enumerate(settings.datasets_names):
        data = np.genfromtxt(os.path.join(settings.results_dir, f"{dataset_name}_evaluation.csv"), delimiter=',',
                             names=True)
        # data = np.genfromtxt(f"result_mix_mutual_info/{dataset_name}_evaluation.csv", delimiter=',', names=True)
        dataset_res = []
        for familiy_id, family_name in enumerate(settings.constr_family_list):
            dataset_res.append(np.mean(data[f"{family_name.replace(' ', '_')}_fscore"]))
            all_res_dict[f"{dataset_name}-{family_name}"] = data[f"{family_name.replace(' ', '_')}_fscore"]
        res_matrix.append(dataset_res)

    # Save results in a table
    with open(os.path.join(settings.output_dir, 'all_results.csv'), mode='w') as ofile:
        writer = csv.writer(ofile, delimiter=',')
        header = ['Dataset'] + [f"{fam.capitalize()}" for fam in list(settings.constr_family_list)]
        writer.writerow(header)
        for dataset_id, dataset_name in enumerate(list(settings.datasets_names)):
            writer.writerow([f"${settings.datasets_labels[dataset_name]}$"] + np.round(100*np.array(res_matrix[dataset_id]), 2).tolist())

    # Compute constraint distribution and plot histogram
    constr_distrib = []
    remove_perfect_dataset = True
    for dataset_id, dataset_name in enumerate(list(settings.datasets_names)):
        if remove_perfect_dataset:
            if np.all(np.array(res_matrix[dataset_id]) == 1):
                continue
        best_constr = np.argwhere(res_matrix[dataset_id] == np.amax(res_matrix[dataset_id]))
        constr_distrib += best_constr.flatten().tolist()

    if remove_perfect_dataset:
        res_matrix = np.delete(res_matrix, (4, 6, 11), axis=0)

    plt.figure()
    plt.rcParams['text.usetex'] = True
    plt.xlabel('Constraint family', fontsize=24)
    plt.xticks([constr_fam for constr_fam in range(len(list(settings.constr_family_list)))],
               labels=[v for k, v in settings.method_label.items()], fontsize=14)
    plt.yticks(fontsize=14)
    plot = sns.histplot(constr_distrib, discrete=True, bins=len(list(settings.constr_family_list)), color="blue")
    plot.set_ylabel("Count", fontsize=20)
    plot.set_ylim([0, 15])
    plt.tight_layout()
    plt.savefig(os.path.join(settings.output_dir, f'constr_histrogram.pdf'))
    plt.rcParams['text.usetex'] = False

    # Ranking for CD diagram
    ranked_res = ss.rankdata(1 - np.array(res_matrix), axis=1, method="min")
    averaged_ranked_res = np.mean(ranked_res, axis=0)
    np.savetxt(os.path.join(settings.output_dir, 'averaged_ranked_res.csv'), averaged_ranked_res, delimiter=',')

    # Plot best results
    dataset_batches = 2
    output_dir = settings.output_dir
    ncols = 3
    nrows = 4

    for batch in range(dataset_batches):
        fig, axs = plt.subplots(nrows, ncols, figsize=(nrows * 3.58, ncols * 4.54), tight_layout=True)
        if batch == 1:
            dataset_batch_list = enumerate(settings.datasets_names[:12])
        else:
            dataset_batch_list = enumerate(settings.datasets_names[12:])
        for dataset_id, dataset_name in dataset_batch_list:
            ax = axs[dataset_id // ncols][dataset_id % ncols]
            ax.set_title(settings.datasets_labels[dataset_name], fontsize=20)
            plt.rcParams['text.usetex'] = True
            for family in settings.constr_family_list:
                prefix_length_list = [e + 1 for e in
                                      range(len(all_res_dict[f"{dataset_name}-{family}"]))]

                ax.plot(prefix_length_list, all_res_dict[f"{dataset_name}-{family}"], ls=settings.method_style[family],
                        color=settings.method_color[family], marker=settings.method_marker[family],
                        label=settings.method_label[family])
                ax.tick_params(axis='both', which='major', labelsize=12)
                ax.tick_params(axis='both', which='minor', labelsize=12)

            plt.rcParams['text.usetex'] = False
            ax.grid()
        if batch == 2:
            axs[-1, -1].axis('off')
            axs[-1, -2].axis('off')  # (0, 1.02, 1.0, 0.2)
        box = axs[0][1].get_position()
        axs[0][1].set_position([box.x0, box.y0, box.width * 0.8, box.height])
        axs[0][1].legend(bbox_to_anchor=(-0.3, 1.02, 1.5, 0.102), loc="lower left", ncol=5, mode="expand", fontsize=17,
                         title="Constraint families", markerscale=2.7, borderaxespad=1.5, fancybox=True, shadow=True,)
        fig.subplots_adjust(top=0.55)
        fig.supylabel('F-score', fontsize=24)
        fig.supxlabel('Prefix length', fontsize=24)
        if batch == 1:
            final_res_name = 'fscore_datasets_0-11.pdf'
        else:
            final_res_name = 'fscore_datasets_11-21.pdf'
        fig.savefig(os.path.join(output_dir, final_res_name))