import matplotlib.pyplot as plt
import os
import pdb


class PlotResult:

    def __init__(self, dict_results, folder):
        self.dict_results = dict_results
        self.folder = folder
        self.method_label = {'existence': r'$\mathrm{CluMeanR}$', 'choice': r'$\mathrm{CluRFMeanR}$',
                             'positive relations': r'$\mathrm{MeanR}$', 'negative relations': r'$\mathrm{RandR}$',
                             'SuppVecMach': r'$\mathrm{SVR}$', 'GradientBoost': r'$\mathrm{GBOOST}$'}
        self.method_marker = {'existence': '<', 'choice': 'x', 'positive relations': 's',
                              'negative relations': '>', 'SuppVecMach': 'D', 'GradientBoost': '^'}
        self.method_color = {'existence': 'mediumpurple', 'choice': 'deepskyblue',
                             'positive relations': 'orange', 'negative relations': 'crimson', 'SuppVecMach': 'forestgreen',
                             'GradientBoost': 'blue'}

    def toPng(self, metric, title):
        plt.clf()
        plt.xlabel('Prefix length', fontsize=18)
        for family_constr in self.dict_results.keys():
            result_list = [getattr(res_obj, metric) for res_obj in self.dict_results[family_constr]]
            plt.plot(range(len(result_list)), result_list,
                     color=self.method_color[family_constr],
                     marker=self.method_marker[family_constr],
                     label=family_constr)
        plt.legend(fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder, f'{title}.pdf'))