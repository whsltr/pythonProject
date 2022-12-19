import pywt
import numpy as np
import matplotlib.pyplot as plt
from spacepy import pycdf
import os
os.environ["CDF_LIB"] = "~/CDF/lib"

cdf = pycdf.CDF('/home/ck/Downloads/omni_hro_5min_20201101_v01.cdf')
print(cdf)