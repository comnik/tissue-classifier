import csv
import datetime
from itertools              import chain

import numpy                as np
import matplotlib.pyplot    as plt

from   sklearn                   import linear_model, metrics, cross_validation, grid_search, multiclass, \
                                        neighbors, ensemble, preprocessing
from   sklearn.preprocessing     import PolynomialFeatures
from   sklearn.svm               import SVR, SVC
from   sklearn.feature_selection import RFE


print "imports loaded"

def scorer(y_true,y_pred):
    """
    Computes the score as detailed in the assignment.
    """
    count = 0
    i = 0
    while i != y_true.shape[0]:
        if y_true[i][0] != y_pred[i][0]:
            count = count + 1

        if y_true[i][1] != y_pred[i][1]:
            count = count + 1
        i = i + 1

    return np.divide(float(count),float(2*y_true.shape[0]))


def to_feature_vec(row):
    """
    Returns the feature-vector representation of a piece of input data.
    """

    poly = PolynomialFeatures(degree=2)
    A,B,C,D,E,F,G,H,I = row[0:9]
    K = row[9:13]
    L = row[13:]
    return [A,C,D,E,F,G,I] + K + L + [i for s in poly.fit_transform([float(A),float(I), float(E)]) for i in s] # removed H, B --> 0.16
    # return [A, E, I, F, G] + K + L 
   


def get_features(inpath):
    """
    Reads our input data from a csv file.
    """
    with open(inpath, 'r') as fin:
        reader = csv.reader(fin, delimiter=',')
        X = [to_feature_vec(row) for row in reader]

    return np.atleast_1d(X)

def make_rfc(num_estimators):
    """
    Returns a random forest classifier.
    """
    return ensemble.RandomForestClassifier(n_estimators=num_estimators, criterion='entropy', n_jobs=-1)



def main():
    plt.ion()

    # Read labelled training data.
    # We train on the tuples (x, y).
    X = get_features('project_data/train.csv')
    Y = np.genfromtxt('project_data/train_y.csv', delimiter=',')

    Xtrain = X
    Ytrain = Y

    # num_neighbors = 4
    # kneigh = neighbors.KNeighborsClassifier(n_neighbors=num_neighbors, weights='distance', p=1)
    # print num_neighbors, "neighbors"

    num_estimators = 100
    rfc = ensemble.RandomForestClassifier(n_estimators=num_estimators, criterion='entropy', min_samples_split=2, n_jobs=-1)
    rfc.fit(Xtrain, Ytrain)
    print "Random Forest with", num_estimators, "estimators."



    # Hplot = Xtrain[:, 0]
    # Xplot = np.atleast_1d([[x] for x in Hplot])
    # Xplot = Xtrain[:, 0]
    # Yplot = kneigh.predict(Xtrain)


    # plt.plot(Xplot, Ytrain, 'bo') # input data
    # plt.plot(Xplot, Yplot, 'ro', linewidth = 3) # prediction
    # plt.plot(Xtrain[:, 0], Xtrain[:, 7], 'bo')
    # plt.show()

    scorefun = metrics.make_scorer(scorer)


    # grid = range(40,125,5)
    # for i in grid:
    #     scores = cross_validation.cross_val_score(make_rfc(i), X, Y, scoring=scorefun, cv = 5)
    #     print('#estimators:', i, 'Mean: ', np.mean(scores), ' +/- ', np.std(scores))


    # scores = cross_validation.cross_val_score(rfc, X, Y, scoring=scorefun, cv = 5)
    # print('Scores: ', scores)
    # print('Mean: ', np.mean(scores), ' +/- ', np.std(scores))

    # regressor_ridge = linear_model.Ridge()
    # param_grid = {'alpha' : np.linspace(0, 100, 10)} # number of alphas is arbitrary
    # n_scorefun = metrics.make_scorer(lambda x, y: -least_squares_loss(x,y)) #logscore is always maximizing... but we want the minium
    # gs = grid_search.GridSearchCV(regressor_ridge, param_grid, scoring = n_scorefun, cv = 5)
    # gs.fit(Xtrain, Ytrain)

    # # print(gs.best_estimator_)
    # # print(gs.best_score_)

    # scores = cross_validation.cross_val_score(gs.best_estimator_, X, Y, scoring = scorefun, cv = 5)
    # print('Scores: ', scores)
    # print('Mean: ', np.mean(scores), ' +/- ', np.std(scores))

    Xval = get_features('project_data/validate.csv')
    # Ypred = gs.best_estimator_.predict(Xval)
    Ypred = rfc.predict(Xval)
    # print(Ypred)
    np.savetxt('out/validate_y.csv', Ypred, delimiter=",", fmt="%i") # the last parameter converts the floats to ints

    # raw_input('Press any key to exit...')


if __name__ == "__main__":
    main()
