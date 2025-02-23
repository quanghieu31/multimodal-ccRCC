"""
Baseline CoxPH models or unimodal Cox models
1) clinical data, linear CoxPH
2) gene expression data, PCA down to 64 components, linear CoxPH
3) WSI images, average pooling for each patches, PCA, linear CoxPH
"""

from lifelines.utils import concordance_index
from lifelines import CoxPHFitter
import numpy as np
import pandas as pd

