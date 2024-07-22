# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 10:11:07 2024

@author: dan
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, TextArea
import numpy as np
import os
import seaborn as sns
import ast
from PIL import Image
import more_plotting_functions as more_plotting

alpha= 0.1

file_name= "optimiser_more_functinos.csv"
df = pd.read_csv(file_name)

"Do stuff"