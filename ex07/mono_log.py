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

def loading(zipcode):
    try:
        bio_data = pd.read_csv("solar_system_census.csv", dtype=np.float64)[['weight', 'height', 'bone_density']].values
        citizens = pd.read_csv("solar_system_census_planets.csv", dtype=np.float64)[['Origin']].values.reshape(-1, 1)
    except Exception as e:
        print(e)
        sys.exit(1)

    # the zipcode is selected ==> 1 other in 0
    for zip in citizens:
        if zip[0] == zipcode:
            zip[0] = 1
        else:
            zip[0] = 0
    
    # split data
    x_train, x_test, y_train, y_test = data_spliter(bio_data, citizens, 0.8)

    #normalizer
    scaler_x = Normalizer(x_train)
    x_train_ = scaler_x.norme(x_train)
    x_test_ = scaler_x.norme(x_test)

    #logistic regression
    print(f"Training to Citizens of {green}{planets[zipcode]}{reset}...")
    thetas = np.array(np.ones(4)).reshape(-1, 1)
    mylr = MyLR(thetas, alpha=0.1, max_iter=50000)
    mylr.fit_(x_train_, y_train)
    
    y_hat = np.around(mylr.predict_(x_test_))
    error = np.mean(y_test != y_hat) * 100
    
    fig, axis = plt.subplots(3, 1)
    fig.suptitle("Error = {:.2f}%".format(error), fontsize=14)
    fig.text(0.04, 0.5, planets[zipcode], va='center', rotation='vertical')
    for idx, i in enumerate(["weight", "height", "bone_density"]):
        axis[idx].scatter(x_test[:, idx], y_test, c="b", marker='o', label="true value")
        axis[idx].scatter(x_test[:, idx], y_hat, c="r", marker='x', label="predicted value")
        axis[idx].set_xlabel(i)
        axis[idx].legend()
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # colors = {1: "b", 0: "r"}
    colors_true = ["b" if i==1 else "r" for i in y_test]
    colors_predict = ["g" if i==1 else "r" for i in y_hat]
    ax.scatter(x_test[:, 0], x_test[:, 1], x_test[:, 2], c=colors_true, marker='o', label="true value")
    ax.scatter(x_test[:, 0], x_test[:, 1], x_test[:, 2], c=colors_predict, marker="x", label="predicted value")
    ax.set_xlabel('weight')
    ax.set_ylabel('height')
    ax.set_zlabel('bone_density')
    ax.set_title("Error = {:.2f}%".format(error))
    ax.legend()
    plt.show()
    
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
    loading(zipcode)
if __name__ == "__main__":
    main()
    print("good bye !")