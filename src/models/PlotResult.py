import pandas as pd
import matplotlib.pyplot as plt
import os
import settings


class PlotResult:

    def __init__(self, dict_results, prefix_length_list, folder):
        self.dict_results = dict_results
        self.folder = folder
        self.prefix_length_list = prefix_length_list

    def toPng(self, metric, title):
        plt.clf()
        if metric == "prec-rec":
            # plt.ylim(0.0, 1.05)
            plt.xlabel('Precision', fontsize=18)
            for family_constr in self.dict_results.keys():
                prec = [getattr(res_obj, "precision") for res_obj in self.dict_results[family_constr]]
                rec = [getattr(res_obj, "recall") for res_obj in self.dict_results[family_constr]]
                plt.plot(rec, prec, color=settings.method_color[family_constr],
                         marker=settings.method_marker[family_constr], label=family_constr)
        else:
            # plt.ylim(0.0, 1.05)
            plt.xlabel('Prefix length', fontsize=18)
            for family_constr in self.dict_results.keys():
                result_list = [getattr(res_obj, metric) for res_obj in self.dict_results[family_constr]]
                plt.plot(self.prefix_length_list, result_list,
                         color=settings.method_color[family_constr],
                         marker=settings.method_marker[family_constr],
                         label=family_constr)
                if metric == "gain":
                    plt.axhline(y=1, color='k', linestyle='--')
        plt.legend(fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder, f'{title}.pdf'))

    @staticmethod
    def plot_time_performance(datasets_list):
        df = pd.DataFrame()
        for dataset in datasets_list:
            tmp_df = pd.read_csv(os.path.join(settings.results_dir, f"{dataset}_recommendation_times.csv"))
            df = df.append(tmp_df)
        df["total_recommendation_time"] = 1000 * df["total_recommendation_time"]

        aggregated_results = df.groupby(['constr_family', 'prefix_length'],
                                        as_index=False)['total_recommendation_time'].aggregate("median")

        plt.clf()
        plt.xlabel('Prefix length', fontsize=18)
        plt.ylabel('Avg. time [ms]', fontsize=18)
        for family_constr in aggregated_results['constr_family'].unique():
            times_per_prefix = aggregated_results[aggregated_results['constr_family'] == family_constr]
            plt.plot(times_per_prefix['prefix_length'], times_per_prefix['total_recommendation_time'],
                     color=settings.method_color[family_constr], marker=settings.method_marker[family_constr],
                     label=family_constr)
        plt.legend(fontsize=14)
        plt.tight_layout()
        if len (datasets_list) > 1:
            title = "aggregated_recommendation_times"
        else:
            title = f"{datasets_list[0]}_recommendation_times"
        plt.savefig(os.path.join(settings.results_dir, f'{title}.pdf'))
