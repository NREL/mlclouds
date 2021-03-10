"""
Make plots of the statistics
"""
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plot_dir = './plots'

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

df = pd.read_csv('./outputs/validation_stats.csv')

variables = df.Variable.unique()
conditions = df.Condition.unique()
metrics = ('MAE (%)', 'MBE (%)', 'RMSE (%)')
models = ('Baseline', 'PHYGNN')

assert all([m in df for m in metrics]), 'Could not find: {}'.format(metrics)
assert all([m in df.Model.unique() for m in models]), 'Could not find: {}'.format(models)

for var in variables:
    for condition in conditions:
        for metric in metrics:
            mask = ((df.Variable == var)
                    & (df.Condition == condition)
                    & df.Model.isin(models))
            df_plot = df[mask]
            sns.barplot(x='Site', y=metric, hue='Model', data=df_plot, ci=None)
            fname = 'stats_{}_{}_{}.png'.format(metric, var, condition)
            fname = fname.lower().replace(' (%)', '')
            fname = fname.replace('-', '_').replace(' ', '_')
            plt.title(fname.replace('.png', ''))
            fp = os.path.join(plot_dir, fname)
            plt.savefig(fp)
            print('Saved: {}'.format(fname))
            plt.close()
