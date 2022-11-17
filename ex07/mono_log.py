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


green = '\033[92m' # vert
blue = '\033[94m' # blue
yellow = '\033[93m' # jaune
red = '\033[91m' # rouge
reset = '\033[0m' #gris, couleur normale
planets = ['The flying cities of Venus', 'United Nations of Earth', 'Mars Republic', 'The Asteroids’ Belt colonies']

def usage():
    print("USAGE:")
    print("\tpython mono.py --zipcode=X\n\t\twith X being 0, 1, 2 or 3\n")
    sys.exit(1)

def loading():
    try:
        bio_data = pd.read_csv("solar_system_census.csv", dtype=np.float64)[['weight', 'height', 'bone_density']].values
        citizens = pd.read_csv("solar_system_census_planets.csv", dtype=np.float64)[['Origin']].values.reshape(-1, 1)
    except Exception as e:
        print(e)
        sys.exit(1)

    return bio_data, citizens

def graphic(x_test, y_test, y_hat, zipcode):
    error = np.mean(y_test != y_hat) * 100
    fig, axis = plt.subplots(3, 1, figsize=(15, 8))
    fig.suptitle("Succes = {:.2f}% - Error = {:.2f}%".format(100-error, error), fontsize=12)
    fig.text(0.04, 0.5, planets[zipcode], va='center', rotation='vertical')
    y_test_text = ['Citizen' if citizen == 1 else 'Foreigner' for citizen in y_test]
    y_hat_text = ['Citizen' if citizen == 1 else 'Foreigner' for citizen in y_hat]
    
    for idx, i in enumerate(["weight", "height", "bone_density"]):
        axis[idx].scatter(x_test[:, idx], y_test_text, c="b", marker='o', label="true value")
        axis[idx].scatter(x_test[:, idx], y_hat_text, c="r", marker='x', label="predicted value")
        axis[idx].set_xlabel(i)
        axis[idx].set_ylabel('')
        axis[idx].legend()
    plt.subplots_adjust(left=0.17,
                    bottom=0.112, 
                    right=0.97, 
                    top=0.933, 
                    wspace=0.195, 
                    hspace=0.291)
    plt.rcParams["figure.figsize"] = (32, 32)
    plt.show()

def citizens_filtered(citizens, zipcode):
    """ return the citizen array with origin = 1 if zipCode 0 otherwise"""
    z = np.copy(citizens)
    for zip in z:
        if zip[0] == zipcode:
            zip[0] = 1
        else:
            zip[0] = 0
    return z

def learning(citizens, bio_data, zipcode, graphics = False):
    """ main learning loop
        args:
            citizens = array filtered for zipcode
            bio_data = array of biologics data
            zipcode = code of the planet
            graphics Boolean to show or not the graphics 
        return:
            a MylR trained
    """
     # split data
    x_train, x_test, y_train, y_test = data_spliter(bio_data, citizens, 0.8)

    #normalizer
    scaler_x = Normalizer(x_train)
    x_train_ = scaler_x.norme(x_train)
    x_test_ = scaler_x.norme(x_test)

    #logistic regression
    print(f"Training to Citizens of '{green}{planets[zipcode]}{reset}' ...")
    thetas = np.array(np.ones(4)).reshape(-1, 1)
    mylr = MyLR(thetas, alpha=0.1, max_iter=50000)
    mylr.fit_(x_train_, y_train)
    
    y_hat = np.around(mylr.predict_(x_test_))
    if graphics:
        graphic(x_test, y_test, y_hat, zipcode)
    return mylr

def compute(bio_data, citizens, zipcode, graphics = False):

    # the zipcode is selected ==> 1 other in 0
    citizens = citizens_filtered(citizens, zipcode)

    #training loop
    return learning(citizens, bio_data, zipcode, graphics)

    
def main():
    try:
        opts, _ = getopt.getopt(sys.argv[1:], "", ['zipcode='])
        if len(opts) != 1 or opts[0][0] != '--zipcode':
            usage()
        zipcode = int(opts[0][1])

    except getopt.GetoptError as inst:
        print(inst)
        usage()
    except ValueError:
        print("zipcode must be an int")
        usage()
    if zipcode not in (0, 1, 2, 3):
        usage()
    bio_data, citizens = loading()
    compute(bio_data, citizens, zipcode, graphics = True)
    
if __name__ == "__main__":
    main()
    print("good bye !")