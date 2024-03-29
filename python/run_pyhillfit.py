from pyhillfit import PyHillFit
import matplotlib.pyplot as plt
#import argparse

data_file = "../data/crumb_data.csv"
drug = "Amiodarone"
channel = "Kv4.3"
fix_hill = True
plot_sigma = False

phf = PyHillFit(data_file, drug, channel, fix_hill=fix_hill)

phf.simple_best_fit_sum_of_squares()  # might have this run automatically, since we'll run it either to plot best fit or to find initial MCMC value

#fig = phf.plot_best_fit(plot_sigma)
#plt.show()

phf.do_adaptive_mcmc()

print phf._chain
