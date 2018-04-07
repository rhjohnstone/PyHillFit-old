import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
#import seaborn as sns
#sns.set()
import cma


#### THINGS TO DO
#
# Do MCMC (single-level and hierarchical, eventually)
#
####


def pic50_to_ic50(pic50): # IC50 in uM
    return 10**(6-pic50)

    
def ic50_to_pic50(ic50): # IC50 in uM
    return 6-np.log10(ic50)


def per_cent_block(conc, ic50, hill=1):
    return 100. * ( 1. - 1./(1.+((1.*conc)/ic50)**hill) )


class PyHillFit(object):

    def __init__(self, data_file, drug, channel, fix_hill=False, mcmc_iterations=100000, pic50_lower_bound=-3):
        all_data = pd.read_csv(data_file)
        self._data = all_data[(all_data["Compound"]==drug) & (all_data["Channel"]==channel)]
        self._drug = drug
        self._channel = channel
        self.fix_hill = fix_hill
        self.mcmc_iterations = mcmc_iterations
        self._pic50_lower_bound = pic50_lower_bound
        if fix_hill:
            def scale_params_for_cmaes(params):
                scaled_pic50 = params[0]**2 + pic50_lower_bound
                return [scaled_pic50, 1.]
        else:
            def scale_params_for_cmaes(params):
                scaled_pic50 = params[0]**2 + pic50_lower_bound
                scaled_hill = params[1]**2  # Hill bounded below at 0
                return [scaled_pic50, scaled_hill]
        self.scale_params_for_cmaes = scale_params_for_cmaes

    def sum_of_square_diffs(self, pic50, hill=1):
        model_blocks = per_cent_block(self._data["Dose"], pic50_to_ic50(pic50), hill)
        return np.sum((model_blocks-self._data["Response"])**2)
    
    def simple_best_fit_sum_of_squares(self, x0=None, sigma0=0.1, cma_random_seed=123):
        opts = cma.CMAOptions()
        opts["seed"] = cma_random_seed
        if x0 is None:
            x0 = [2.5, 1.]
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        while not es.stop():
            X = es.ask()
            es.tell(X, [self.sum_of_square_diffs(*self.scale_params_for_cmaes(x)) for x in X])
            es.disp()
        res = es.result
        self._best_fit_params = self.scale_params_for_cmaes(res[0])

    def plot_best_fit(self):
        fig = plt.figure(figsize=(4, 3))
        ax = fig.add_subplot(111)
        xmin_log10 = int(np.log10(self._data["Dose"].min()))-1
        xmax_log10 = int(np.log10(self._data["Dose"].max()))+2
        num_x_pts = 50
        x = np.logspace(xmin_log10, xmax_log10, num_x_pts)
        ax.set_xscale("log")
        pic50, hill = self._best_fit_params
        ax.plot(x, per_cent_block(x, pic50_to_ic50(pic50), hill), label="Best", lw=2)
        ax.plot(self._data["Dose"], self._data["Response"], "o", label="Data", clip_on=False, zorder=10, ms=5)
        ax.legend(loc="best")
        ax.set_xlabel("{} concentration ($\mu$M)".format(self._drug))
        ax.set_ylabel("% {} block".format(self._channel))
        ax.set_xlim(10**xmin_log10, 10**xmax_log10)
        ax.set_ylim(0, 100)
        if self.fix_hill:
            ax.set_title("Vary $pIC_{50}$, fix $Hill=1$")
        else:
            ax.set_title("Vary $pIC_{50}$ and $Hill$")
        fig.tight_layout()
        return fig

