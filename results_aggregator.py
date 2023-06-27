import pdb
import settings
import csv
import os
import numpy as np
import pandas as pd
from src.models.PlotResult import PlotResult


def aggregate_recommendation_results(metric="fscore"):
    """Available metrics: comp, non_comp, pos_non_comp, tp, fp, tn, fn, precision, recall, fscore"""
    with open(os.path.join(settings.results_dir, f"aggregated_recommendation_performance.csv"), mode='w') as out_file:
        writer = csv.writer(out_file, delimiter=',')
        writer.writerow(["Dataset"] + list(settings.constr_family_list))
        for dataset in settings.datasets_names:
            df_results = pd.read_csv(os.path.join(settings.results_dir, f"{dataset}_evaluation.csv"))
            writer.writerow([dataset] + [round(100 * np.mean(df_results[f"{constr_family}_{metric}"]), 2)
                                         for constr_family in settings.constr_family_list])


if __name__ == "__main__":
    # Plot time performance for each single dataset
    for dataset in settings.datasets_names:
        PlotResult.plot_time_performance([dataset])

    # Plot aggregated time performance
    PlotResult.plot_time_performance(settings.datasets_names)

    # Compute and save aggregated recommendation performance
    aggregate_recommendation_results()
