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
from mono_log import loading, citizens_filtered

green = '\033[92m' # vert
blue = '\033[94m' # blue
yellow = '\033[93m' # jaune
red = '\033[91m' # rouge
reset = '\033[0m' #gris, couleur normale
planets = ['The flying cities of Venus', 'United Nations of Earth', 'Mars Republic', 'The Asteroids’ Belt colonies']

def main():
    bio, citi = loading()
    #split
    x_train, x_test, y_train, y_test = data_spliter(bio, citi, 0.8)
    #normalizer
    scaler_x = Normalizer(x_train)
    x_train_ = scaler_x.norme(x_train)
    x_test_ = scaler_x.norme(x_test)
    list_reg = []
    for zipcode in range(4):
        citizens_filtred = citizens_filtered(citizens=y_train, zipcode=zipcode)
        #logistic regression
        print(f"Training to Citizens of '{green}{planets[zipcode]}{reset}' ...")
        thetas = np.array(np.ones(4)).reshape(-1, 1)
        mylr = MyLR(thetas, alpha=0.1, max_iter=50000)
        mylr.fit_(x_train_, citizens_filtred)
        list_reg.append(mylr.predict_(x_test_))
    
    for i in range(len(y_test)):
        print(f"{list_reg[0][i]} - {list_reg[1][i]} - {list_reg[2][i]} - {list_reg[3][i]}")
if __name__ == "__main__":
    main()
    print("good bye !")