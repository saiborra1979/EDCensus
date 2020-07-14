"""
FUNCTION TO IMPLEMENT NON-PARAMETRIC NADARYA-WATSON ESTIMATOR

https://royalsocietypublishing.org/doi/pdf/10.1098/rsta.2011.0550
http://www.gaussianprocess.org/gpml/chapters/RW5.pdf
http://www.cs.cmu.edu/~epxing/Class/10708-16/note/10708_scribe_lecture20.pdf
https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_co2.html
https://nbviewer.jupyter.org/github/SheffieldML/notebook/blob/master/GPy/index.ipynb
"""

import os
import pickle
import numpy as np
import pandas as pd
from time import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

