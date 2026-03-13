import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import scripts.plot as myplt
import scripts.util as util

if __name__ == '__main__':
    
    myplt.plot_init()
    D, L = util.load(sys.argv[1])
    myplt.plot_hist(D, L)
    myplt.plot_scatter(D, L)
    util.calc(D, L)