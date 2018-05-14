# I should really abstract out / modularise the methods used here and in pyhillfit just so they're definitely exactly the same
# The data in some_data.txt is from crumb_data.csv, just in a copy and paste-able format
# This operates on a simple sum-of-squares score, no fancy "folded likelihood" like in my thesis,
# therefore the data should ideally be raw, i.e. no capping at 0 or 100.

### TO DO ###
# 1. Drug and channel name entry boxes and update axis labels
# 2. Add button to save figure (pdf/eps/png?)
# 3. Display best-fit parameter values somewhere, should be easily exportable
# 4. Display sum-of-squares score and/or BIC somewhere
# 5. I actually think that, since I'm building this to compare the two models,
#    that it's not worth trying to keep things "general", and that I should more
#    explicitly define things for the two different models, e.g. log-likelihoods
###

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

def scale_params_for_cmaes_model_1(unscaled_pic50):
    """Bound pIC50 above some value and fix Hill=1"""
    scaled_pic50 = unscaled_pic50**2 + pic50_lower_bound
    return scaled_pic50
    
def scale_params_for_cmaes_model_2(unscaled_pic50, unscaled_hill):
    """Bound pIC50 above some value and bound Hill above 0"""
    scaled_pic50 = unscaled_pic50**2 + pic50_lower_bound
    scaled_hill = unscaled_hill**2  # Hill bounded below at 0
    return [scaled_pic50, scaled_hill]

def compute_best_sigma_analytic(sum_of_squares, num_data_pts):
    """MLE computation of observation noise s.d. sigma"""
    return np.sqrt((1.*sum_of_squares)/num_data_pts)

def compute_max_log_likelihood(sum_of_squares, num_data_pts, best_sigma):
    """Assuming Normal distribution of data points"""
    return -0.5*num_data_pts*np.log(2*np.pi) - num_data_pts*np.log(best_sigma) - sum_of_squares/(2.*best_sigma**2)

def sum_of_square_diffs(concs, responses, pic50, hill=1):
    model_blocks = per_cent_block(concs, pic50_to_ic50(pic50), hill)
    return np.sum((model_blocks-responses)**2)



width, height = 4, 3
fig = Figure(figsize=(width, height))
ax = fig.add_subplot(111)
ax.set_ylim(0,100)
ax.set_xscale("log")
ax.set_xlim(10**-3, 10**3)
ax.set_ylabel("% Block")
ax.set_xlabel("Concentration")
fig.set_tight_layout(True)

fit_m1_text = "Fit M1"
fit_m2_text = "Fit M2"

default_font = "TkDefaultFont 9"

class MyFirstGUI:
    def __init__(self, master):
        self.master = master
        master.title("PyHillGUI")
        
        self.data_label = tk.Label(master, text="Data")
        self.data_label.grid(row=0, column=0, columnspan=2)

        self.text = tk.Text(master, height=16, width=12)
        self.text.grid(column=0, row=1, columnspan=2)
        self.text.insert(tk.END, 'x1, x2, xn\ny1, y2, yn\n\nor\n\nx1, y1\nx2, y2\nxn, yn')
        self.text.bind("<Key>", self.key)

        self.plot_data_button = tk.Button(master, text="Plot data", command=self.read_box)
        self.plot_data_button.grid(column=0, row=2, columnspan=2)#, padx=4)

        self.fit_m1_button = tk.Button(master, text=fit_m1_text, command=self.do_best_fit_model_1)
        self.fit_m1_button.grid(column=0, row=3)#, padx=4)

        self.fit_m2_button = tk.Button(master, text=fit_m2_text, command=self.do_best_fit_model_2)
        self.fit_m2_button.grid(column=1, row=3)#, padx=4)

        self.plot_m1_button = tk.Button(master, text="Plot M1", command=self.plot_m1_fit)
        self.plot_m1_button.grid(column=0, row=4)#, padx=4)

        self.plot_m2_button = tk.Button(master, text="Plot M2", command=self.plot_m2_fit)
        self.plot_m2_button.grid(column=1, row=4)#, padx=4)

        self.close_button = tk.Button(master, text="Close", command=master.quit)
        self.close_button.grid(column=0, row=5, columnspan=2)

        canvas = FigureCanvasTkAgg(fig, master)
        canvas.draw()
        canvas.get_tk_widget().grid(column=2, row=0, rowspan=7)
        
        self.m1_bic_text = tk.StringVar()
        self.m1_bic_label = tk.Label(master, textvariable=self.m1_bic_text)
        self.m1_bic_text.set("M1 BIC\n")
        self.m1_bic_label.grid(column=3, row=2, padx=6)
        
        self.m2_bic_text = tk.StringVar()
        self.m2_bic_label = tk.Label(master, textvariable=self.m2_bic_text)
        self.m2_bic_text.set("M2 BIC\n")
        self.m2_bic_label.grid(column=3, row=3, padx=6)
        
        self.best_m1_pic50, self.best_m1_sigma = None, None
        self.best_m2_pic50, self.best_m2_hill, self.best_m2_sigma = None, None, None
        
        self.m1_ss, self.m2_ss = None, None
        self.bic_1, self.bic_2 = None, None
        
        self.num_data_pts = None
        
    def simple_best_fit_sum_of_squares_model_1(self, x0=None, sigma0=0.1, cma_random_seed=None):
        """Optimisation keeping Hill=1 fixed and varying pIC50 but enforcing its lower bound,
           but CMA-ES operates in minimium 2-d, so have to trick it"""
        opts = cma.CMAOptions()
        if cma_random_seed is not None:
            opts["seed"] = cma_random_seed
        if x0 is None:
            x0 = [2.5, 1.]
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        while not es.stop():
            X = es.ask()
            es.tell(X, [sum_of_square_diffs(self.data[:,0], self.data[:,1], scale_params_for_cmaes_model_1(x[0])) for x in X])
            #es.disp()
        res = es.result
        self.best_m1_sigma = compute_best_sigma_analytic(res[1], self.num_data_pts)
        self.best_m1_pic50 = scale_params_for_cmaes_model_1(res[0][0])
        
        model = 1
        self.bic_1 = self.compute_bic(model, res[1])

    def simple_best_fit_sum_of_squares_model_2(self, x0=None, sigma0=0.1, cma_random_seed=123):
        """Optimisation varying both pIC50 and Hill, but enforcing the lower bounds"""
        opts = cma.CMAOptions()
        #opts["seed"] = cma_random_seed
        if x0 is None:
            x0 = [2.5, 1.]
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        while not es.stop():
            X = es.ask()
            es.tell(X, [sum_of_square_diffs(self.data[:,0], self.data[:,1], *scale_params_for_cmaes_model_2(*x)) for x in X])
            #es.disp()
        res = es.result
        self.best_m2_sigma = compute_best_sigma_analytic(res[1], self.num_data_pts)
        self.best_m2_pic50, self.best_m2_hill = scale_params_for_cmaes_model_2(*res[0])
        
        model = 2
        self.bic_2 = self.compute_bic(model, res[1])
        
    def key(self, event):
        self.plot_data_button.config(font=default_font)
        self.m1_bic_text.set("M1 BIC\n")
        self.m2_bic_text.set("M2 BIC\n")

    def read_box(self):
        self.plot_data_button.config(font=default_font+" overstrike")
        self.fit_m1_button.config(font=default_font)
        self.fit_m2_button.config(font=default_font)
        self.plot_m1_button.config(font=default_font)
        self.plot_m2_button.config(font=default_font)
        self.data = convert_strings_to_array(self.text.get("1.0", tk.END).rstrip())
        self.num_data_pts = len(self.data[:,1])
        plot_data(*self.data.T)
        
        self.m1_line_plotted = False
        self.m1_line = None
        
        self.m2_line_plotted = False
        self.m2_line = None

    def compute_bic(self, model, sum_of_squares):
        if model==1:
            num_params = 2
            best_sigma = self.best_m1_sigma
        elif model==2:
            num_params = 3
            best_sigma = self.best_m2_sigma
        return np.log(self.num_data_pts)*num_params - 2*compute_max_log_likelihood(sum_of_squares, self.num_data_pts, best_sigma)
        
    def do_best_fit_model_1(self):
        self.simple_best_fit_sum_of_squares_model_1()
        self.fit_m1_button.config(font=default_font+" overstrike")
        
    def do_best_fit_model_2(self):
        self.simple_best_fit_sum_of_squares_model_2()
        self.fit_m2_button.config(font=default_font+" overstrike")
        
    def plot_m1_fit(self):
        self.plot_m1_button.config(font=default_font+" overstrike")
        xmin = int(np.log10(self.data[:,0].min()))-1
        xmax = int(np.log10(self.data[:,0].max()))+2
        x = np.logspace(xmin, xmax, n)
        if not self.m1_line_plotted:
            self.m1_line, = ax.plot(x, per_cent_block(x, pic50_to_ic50(self.best_m1_pic50), 1), lw=2, label="Best M1", color="C1")
            self.m1_line_plotted = True
        else:
            self.m1_line.remove()
            self.plot_m1_button.config(font=default_font)
            self.m1_line_plotted = False
        ax.legend(bbox_to_anchor=anchor, loc="upper left")
        fig.canvas.draw()
        self.m1_bic_text.set("M1 BIC\n{}".format(round(self.bic_1,1)))
        
    def plot_m2_fit(self):
        self.plot_m2_button.config(font=default_font+" overstrike")
        xmin = int(np.log10(self.data[:,0].min()))-1
        xmax = int(np.log10(self.data[:,0].max()))+2
        x = np.logspace(xmin, xmax, n)
        if not self.m2_line_plotted:
            self.m2_line, = ax.plot(x, per_cent_block(x, pic50_to_ic50(self.best_m2_pic50), self.best_m2_hill), lw=2, label="Best M2", color="C2")
            self.m2_line_plotted = True
        else:
            self.m2_line.remove()
            self.plot_m2_button.config(font=default_font)
            self.m2_line_plotted = False
        ax.legend(bbox_to_anchor=anchor, loc="upper left")
        fig.canvas.draw()
        self.m2_bic_text.set("M2 BIC\n{}".format(round(self.bic_2,1)))
    
anchor = (0.04, 0.98)

def plot_data(xx, yy):
    ax.cla()
    try:
        ax.legend().set_visible(False)
    except:
        pass
    ax.set_ylim(0,100)
    ax.set_xscale("log")
    ax.set_ylabel("% Block")
    ax.set_xlabel("Concentration")
    xmin = int(np.log10(xx.min()))-1
    xmax = int(np.log10(xx.max()))+2
    x = np.logspace(xmin, xmax, n)
    ax.set_xlim(10**xmin, 10**xmax)
    data_plot = ax.plot(xx, yy, "o", label="Data", clip_on=False, zorder=10, ms=5, color="C0")
    ax.legend(bbox_to_anchor=anchor, loc="upper left")
    #fig.tight_layout()
    fig.canvas.draw()
    
n = 100

if __name__=="__main__":
    root = tk.Tk()
    my_gui = MyFirstGUI(root)
    root.mainloop()
