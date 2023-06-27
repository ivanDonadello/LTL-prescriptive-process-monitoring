import pandas as pd
import matplotlib.pyplot as plt
import os
import settings

dataset_names = ["bpic2011_f1", "bpic2011_f2", "bpic2011_f3", "bpic2011_f4",
                  "bpic2012_accepted", "bpic2012_cancelled", "bpic2012_declined",
                  "bpic2015_1_f2", "bpic2015_2_f2", "bpic2015_3_f2", "bpic2015_4_f2",
                  "bpic2015_5_f2", "hospital_billing_2", "hospital_billing_3", "Production",
                  "sepsis_cases_1", "sepsis_cases_2", "sepsis_cases_4"]

df = pd.DataFrame()
for dataset in dataset_names:
    tmp_df = pd.read_csv(os.path.join(settings.results_dir, f"{dataset}_recommendation_times.csv"))
    df = pd.concat([df, tmp_df])
df["total_recommendation_time"] = 1000*df["total_recommendation_time"]
aggregated_results = df.groupby(['constr_family', 'prefix_length'],
                                as_index=False)['total_recommendation_time'].aggregate('mean')

plt.xlabel('Prefix length', fontsize=18)
plt.ylabel('Avg. time [ms]', fontsize=18)
for family_constr in aggregated_results['constr_family'].unique():
    times_per_prefix = aggregated_results[aggregated_results['constr_family'] == family_constr]
    plt.plot(times_per_prefix['prefix_length'], times_per_prefix['total_recommendation_time'],
             color=settings.method_color[family_constr], marker=settings.method_marker[family_constr],
             label=family_constr)
plt.legend(fontsize=14)
plt.tight_layout()
title = "aggregated_recommendation_times"
plt.savefig(os.path.join(settings.results_dir, f'{title}.pdf'))
