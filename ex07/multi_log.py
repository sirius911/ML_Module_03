import getopt
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_spliter import data_spliter
from Normalizer import Normalizer

path = os.path.join(os.path.dirname(__file__), '..', 'ex06')
sys.path.insert(1, path)
from my_logistic_regression import MyLogisticRegression as MyLR
from mono_log import loading, compute

green = '\033[92m' # vert
blue = '\033[94m' # blue
yellow = '\033[93m' # jaune
red = '\033[91m' # rouge
reset = '\033[0m' #gris, couleur normale
planets = ['The flying cities of Venus', 'United Nations of Earth', 'Mars Republic', 'The Asteroids’ Belt colonies']

def main():
    bio, citi = loading()
    list_reg = []
    for zipcode in range(4):
        list_reg.append(compute(bio_data=bio, citizens=citi, zipcode=zipcode))
    
if __name__ == "__main__":
    main()
    print("good bye !")