import getopt
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_spliter import data_spliter
from Normalizer import Normalizer
from mpl_toolkits.mplot3d import Axes3D

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

def graph(axis, data_graph, dim='2D'):
    if dim not in ('2D', '3D'):
        print("Error in graph: bad dim")
        return
    x = data_graph['x']
    y = data_graph['y']
    if dim == '3D':
        z = data_graph['z']
        z_label = data_graph['z_label']
    x_label = data_graph['x_label']
    y_label = data_graph['y_label']
    true_planet = data_graph['true_planet']
    predicted_planet = data_graph['predicted_value']

    planet_colors = ['r', 'b', 'c', 'y']
    planet_names = [planets[int(i)] for i in true_planet]
    colors_true = [planet_colors[int(i[0])] for i in true_planet]
    colors_predicted = [planet_colors[i] for i in predicted_planet]
    if dim == '2D':
        pdf = pd.DataFrame({'x':x, "y":y, 'colors':colors_true, 'planet':planet_names})
    else:
        pdf = pd.DataFrame({'x':x, "y":y, 'z':z, 'colors':colors_true, 'planet':planet_names})
    for planet, dff in pdf.groupby('planet'):
        if dim == '2D':
            axis.scatter(dff['x'], dff['y'], c=dff['colors'], marker='o', s=20, label=planet)
            axis.scatter(x, y, c=colors_predicted, marker='x', s=40, label=None)
        else:
            axis.scatter(dff['x'], dff['y'], dff['z'], c=dff['colors'], marker='o', label=planet)
            axis.scatter(x, y, z, c=colors_predicted, marker='x', s=40, label=None)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    if dim == '3D':
        axis.set_zlabel(z_label)
    axis.legend()

def best_result(array):
    """ return the num colomun so the num of planet"""
    if len(array) != 4:
        print("Error in best_result(): bad array")
        return None
    column_val = array[0]
    column = 0
    for i in range(1, 4):
        if array[i] > column_val:
            column_val = array[i]
            column = i
    return column


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
        mylr = MyLR(thetas, alpha=0.1, max_iter=5000)
        mylr.fit_(x_train_, citizens_filtred)
        list_reg.append(mylr.predict_(x_test_))

    concat = np.concatenate((list_reg[0], list_reg[1], list_reg[2], list_reg[3]), axis=1)

    
    print("analyze ...")
    predicted_value = []
    score = 0
    for i, (c, y) in enumerate(zip(concat, y_test)):
        print(f"{i}", end='')
        best = best_result(c)
        predicted_value.append(best)
        print(f" -> {planets[best]}", end='')
        if best == y[0]:
            print(f" = {green}{planets[int(y[0])]}{reset}")
            score += 1
        else:
            print(f" != {red}{planets[int(y[0])]}{reset}")
    succes = score / len(y_test) * 100
    error = 100 - succes
    print(f"Score : succes = {yellow}{succes:0.2f}%{reset}\terror = {yellow}{error:0.2f}%{reset}")


    #graph 2D
    fig, axis = plt.subplots(3, 1, figsize=(15, 8))
    fig.suptitle("Succes = {:.2f}% - Error = {:.2f}%".format(succes, error), fontsize=12)

    graph(axis[0], {'x':x_test[:, 0], 'y':x_test[:, 1],
                'x_label': 'Weight', 'y_label':'Height',
                'true_planet':y_test, 'predicted_value':predicted_value,}, dim='2D')

    graph(axis[1], {'x':x_test[:, 0], 'y':x_test[:, 2],
                'x_label': 'Weight', 'y_label':'Bone density',
                'true_planet':y_test, 'predicted_value':predicted_value,}, dim='2D')
    graph(axis[2], {'x':x_test[:, 2], 'y':x_test[:, 1],
                'x_label': 'Bone density', 'y_label':'Height',
                'true_planet':y_test, 'predicted_value':predicted_value,}, dim='2D')
    
    plt.subplots_adjust(left=0.05,
                    bottom=0.08, 
                    right=0.98, 
                    top=0.93, 
                    wspace=0.195, 
                    hspace=0.291)
    #graph 3D
    fig = plt.figure()
    ax = Axes3D(fig)
    graph(ax, {'x':x_test[:, 0], 'y':x_test[:, 1], 'z':x_test[:, 2],
                'x_label': 'Weight', 'y_label':'Height', 'z_label':'Bone density',
                'true_planet':y_test, 'predicted_value':predicted_value,}, dim='3D')
    
    plt.subplots_adjust(left=0.05,
                    bottom=0.08, 
                    right=0.98, 
                    top=0.93, 
                    wspace=0.195, 
                    hspace=0.291)
    plt.show()

if __name__ == "__main__":
    main()
    print("good bye !")