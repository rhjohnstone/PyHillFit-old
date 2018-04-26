# I should really abstract out / modularise the methods used here and in pyhillfit just so they're definitely exactly the same
# Also this currently only works for varying pIC50 and Hill (i.e. not Hill fixed to 1)
# The data in some_data.txt is from crumb_data.csv, just in a copy and paste-able format

import numpy as np
import cma
import sys
if sys.version_info[0] < 3:
    import Tkinter as tk
else:
    import tkinter as tk


import matplotlib
matplotlib.use('TKAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
plt.style.use("ggplot")

def pic50_to_ic50(pic50): # IC50 in uM
    return 10**(6-pic50)

def per_cent_block(conc, ic50, hill=1):
    return 100. * ( 1. - 1./(1.+((1.*conc)/ic50)**hill) )

def convert_strings_to_array(strings):
    """Numbers must be entered into text box already in the correct shape (or transposed), separated by commas"""
    row_strings = strings.split("\n")
    new_array = np.array([[float(i) for i in row_string.split(",")] for row_string in row_strings])
    shape = new_array.shape
    if shape[1]==2:
        return new_array
    elif shape[0]==2:
        return new_array.T
    else:
        print "Currently only accepting arrays of shape (2,x) or (x,2)"
        return None

pic50_lower_bound = -2
def scale_params_for_cmaes(params):
    scaled_pic50 = params[0]**2 + pic50_lower_bound
    scaled_hill = params[1]**2  # Hill bounded below at 0
    return [scaled_pic50, scaled_hill]

def compute_best_sigma_analytic(sum_of_squares, num_data_pts):
    return np.sqrt((1.*sum_of_squares)/num_data_pts)

width, height = 4, 3
fig = Figure(figsize=(width, height))
ax = fig.add_subplot(111)
ax.set_ylim(0,100)
ax.set_xscale("log")
ax.set_xlim(10**-3, 10**3)
ax.set_ylabel("% Block")
ax.set_xlabel("Concentration")
fig.set_tight_layout(True)

class MyFirstGUI:
    def __init__(self, master):
        self.master = master
        master.title("PyHillGUI")

        self.text = tk.Text(master, height=16, width=10)
        self.text.grid(column=0)
        self.text.insert(tk.END, 'Data here')

        self.read_box_button = tk.Button(master, text="Fit and plot", command=self.read_box)
        self.read_box_button.grid(column=0, padx=4)

        self.close_button = tk.Button(master, text="Close", command=master.quit)
        self.close_button.grid()

        canvas = FigureCanvasTkAgg(fig, master)
        canvas.draw()
        canvas.get_tk_widget().grid(column=1, row=0, rowspan=3)
        
        self.num_params = 3  # pIC50, Hill, and sigma
    


    def read_box(self):
        self.data = convert_strings_to_array(self.text.get("1.0", tk.END).rstrip())
        new_best_params = simple_best_fit_sum_of_squares(self.data[:,0], self.data[:,1])
        update_plot(new_best_params, *self.data.T)
        
def sum_of_square_diffs(concs, responses, pic50, hill=1):
    model_blocks = per_cent_block(concs, pic50_to_ic50(pic50), hill)
    return np.sum((model_blocks-responses)**2)

def simple_best_fit_sum_of_squares(concs, responses, x0=None, sigma0=0.1, cma_random_seed=123):
    opts = cma.CMAOptions()
    opts["seed"] = cma_random_seed
    if x0 is None:
        x0 = [2.5, 1.]
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    while not es.stop():
        X = es.ask()
        es.tell(X, [sum_of_square_diffs(concs, responses, *scale_params_for_cmaes(x)) for x in X])
        #es.disp()
    res = es.result
    best_sigma = compute_best_sigma_analytic(res[1], len(responses))
    #best_fit_params = np.concatenate((scale_params_for_cmaes(res[0]), [best_sigma]))
    return scale_params_for_cmaes(res[0])

n = 100
def update_plot(new_params, xx, yy):
    ax.cla()
    ax.set_ylim(0,100)
    ax.set_xscale("log")
    ax.set_ylabel("% Block")
    ax.set_xlabel("Concentration")
    xmin = int(np.log10(xx.min()))-1
    xmax = int(np.log10(xx.max()))+2
    x = np.logspace(xmin, xmax, n)
    ax.set_xlim(10**xmin, 10**xmax)
    pic50, hill = new_params  # only works for varying Hill currently
    ax.plot(x, per_cent_block(x, pic50_to_ic50(pic50), hill), lw=2, label="Best fit")
    ax.plot(xx, yy, "o", label="Data", clip_on=False, zorder=10, ms=5)
    fig.legend(loc=(0.21, 0.75))
    #fig.tight_layout()
    fig.canvas.draw()
        

if __name__=="__main__":
    root = tk.Tk()
    my_gui = MyFirstGUI(root)
    root.mainloop()
