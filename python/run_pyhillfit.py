from pyhillfit import PyHillFit
import matplotlib.pyplot as plt
#import argparse

data_file = "../data/crumb_data.csv"
drug = "Amiodarone"
channel = "hERG"
fix_hill = False

phf = PyHillFit(data_file, drug, channel, fix_hill=fix_hill)
phf.simple_best_fit_sum_of_squares()  # might have this run automatically, since we'll run it to plot best fit and to find initial MCMC value
fig = phf.plot_best_fit()
plt.show()

