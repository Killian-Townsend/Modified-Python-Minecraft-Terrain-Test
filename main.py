# Imports
import sys, getopt
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import Voronoi
from skimage.draw import polygon
from PIL import Image
from noise import snoise3


# Modules
import generate


# Variables
size = 1024
n=256
map_seed = 762345


generate.init(map_seed,size,n)
