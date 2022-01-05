import matplotlib.pyplot as plt
import settings
from src.models.EvaluationResult import EvaluationResult
import os
import csv
import pdb
import seaborn as sns
import numpy as np

np. set_printoptions(suppress=True)
plt.rc('font', family='serif')

if __name__ == "__main__":
    bins = [e for e in range(5, 100, 10)]
    th_distrib = []
    all_res_dict = {}

    # Gather results from single files
    for th in bins:
        dir_res = f"src/machine_learning/cum_75CO_mean_{th}_gain_scoreW1"

        with open(os.path.join(dir_res, 'results.csv')) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            next(csv_reader)

            for row in csv_reader:
                dataset = row[0]
                if dataset is not "":
                    for family_id, family in enumerate(settings.constr_family_list):
                        ev_res = EvaluationResult()
                        ev_res.th = int(th)
                        ev_res.fscore = float(row[family_id + 1])

                        eval_data = np.genfromtxt(os.path.join(dir_res, 'result', f'{dataset}_evaluation.csv'),
                                                  names=True, delimiter=',')
                        ev_res.fscore_list = eval_data[f"{family.replace(' ', '_')}_fscore"]

                        if f"{dataset}-{family}" in all_res_dict.keys():
                            all_res_dict[f"{dataset}-{family}"].append(ev_res)
                        else:
                            all_res_dict[f"{dataset}-{family}"] = [ev_res]
    tmp_dict = {}
    constr_distrib = []

    # res_matrix = [['']*len(settings.constr_family_list)]*len(settings.datasets_names)

    res_matrix = np.zeros(shape=(len(settings.datasets_names), len(settings.constr_family_list)), dtype=np.float32)
    th_matrix = np.zeros(shape=(len(settings.datasets_names), len(settings.constr_family_list)), dtype=np.int32)

    for k, v in all_res_dict.items():
        tmp_list = sorted(v, key=lambda eval_res: (- eval_res.fscore), reverse=False)
        all_res_dict[k] = tmp_list
        th_distrib.append(tmp_list[0].th)
        dataset_tmp = k.split("-")[0]
        cons_fam_tmp = k.split("-")[1]
        constr_index = list(settings.constr_family_list).index(cons_fam_tmp)
        dataset_index = list(settings.datasets_names).index(dataset_tmp)
        res_matrix[dataset_index][constr_index] = tmp_list[0].fscore
        th_matrix[dataset_index][constr_index] = tmp_list[0].th

    # Plot best results
    ncols = 3
    nrows = 4
    fig, axs = plt.subplots(nrows, ncols, figsize=(nrows*3.58, ncols*4.54), tight_layout=True)

    for dataset_id, dataset_name in enumerate(settings.datasets_names[12:]):
        ax = axs[dataset_id // ncols][dataset_id % ncols]
        ax.set_title(dataset_name)
        plt.rcParams['text.usetex'] = True
        for family in settings.constr_family_list:
            prefix_lenght_list = [e + 1 for e in
                                  range(len(all_res_dict[f"{dataset_name}-{family}"][0].fscore_list))]

            ax.plot(prefix_lenght_list, all_res_dict[f"{dataset_name}-{family}"][0].fscore_list,
                    color=settings.method_color[family], marker=settings.method_marker[family],
                    label=settings.method_label[family])

        plt.rcParams['text.usetex'] = False
        ax.grid()
    axs[-1, -1].axis('off')
    axs[-1, -2].axis('off')
    axs[0][1].legend(bbox_to_anchor=(0, 1.02, 1.0, 0.2), loc="lower left", ncol=5, mode="expand", fontsize='large',
                     title="Constraint families", borderaxespad=1.5)
    fig.subplots_adjust(top=0.55)
    fig.supylabel('F-score', fontsize='x-large')
    fig.supxlabel('Prefix lenght', fontsize='x-large')
    fig.savefig(os.path.join(settings.output_dir, f'fscore_datasets_11-21.pdf'))

    """
    for dataset_name in settings.datasets_names:
        plt.clf()
        plt.xlabel('Prefix length', fontsize=18)
        plt.title(dataset_name)
        plt.rcParams['text.usetex'] = True
        for family in settings.constr_family_list:
            prefix_lenght_list = [e + 1 for e in range(len(all_res_dict[f"{dataset_name}-{family}"][0].fscore_list))]
            plot_res = PlotResult(None, None, None)
            plt.plot(prefix_lenght_list, all_res_dict[f"{dataset_name}-{family}"][0].fscore_list,
                     color=plot_res.method_color[family], marker=plot_res.method_marker[family],
                     label=plot_res.method_label[family])
        plt.legend(fontsize=14, ncol=2)
        plt.tight_layout()
        plt.rcParams['text.usetex'] = False
        plt.savefig(os.path.join(settings.output_dir, f'fscore_{dataset_name}.pdf'))
    """

    # Plot constr threshold histogram
    plt.clf()
    plt.figure()
    plt.rcParams['text.usetex'] = False
    plt.xlabel('Threshold percentage values', fontsize='x-large')
    plt.xticks([th for th in bins])
    sns.histplot(th_distrib, discrete=True, bins=len(bins), color="blue")
    plt.tight_layout()
    plt.savefig(os.path.join(settings.output_dir, f'th_histrogram.pdf'))

    # Compute constraint distribution
    for dataset_res in res_matrix:
        best_constr = np.argwhere(dataset_res == np.amax(dataset_res))
        constr_distrib += best_constr.flatten().tolist()

    # Plot constraint histogram
    plt.figure()
    plt.rcParams['text.usetex'] = True
    plt.xlabel('Constraint family', fontsize='x-large')
    plt.xticks([constr_fam for constr_fam in range(len(list(settings.constr_family_list)))],
               labels=[v for k, v in settings.method_label.items()])
    sns.histplot(constr_distrib, discrete=True, bins=len(list(settings.constr_family_list)), color="blue")
    plt.tight_layout()
    plt.savefig(os.path.join(settings.output_dir, f'constr_histrogram.pdf'))
    plt.rcParams['text.usetex'] = False

    # Export numeric results
    with open(os.path.join(settings.output_dir, 'all_results.csv'), mode='w') as ofile:
        writer = csv.writer(ofile, delimiter=',')
        writer.writerow(['Dataset'] + list(settings.constr_family_list))
        for dataset_id, dataset_name in enumerate(list(settings.datasets_names)):
            writer.writerow([dataset_name] + [f"{score:.2f} ({th_matrix[dataset_id][score_id]})" for score_id, score in
                                              enumerate(res_matrix[dataset_id])])
