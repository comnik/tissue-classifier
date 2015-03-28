import csv
import datetime

import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing, metrics, cross_validation, grid_search, neighbors, ensemble


def scorer(y_true, y_pred):
    """
    Computes the score as detailed in the assignment.
    """

    n              = y_true.shape[0]
    count_errors   = lambda truth, pred: [(truth[0] != pred[0]), (truth[1] != pred[1])].count(True)
    misclassifieds = [count_errors(truth, pred) for truth, pred in zip(y_true, y_pred)]
    count          = sum(misclassifieds)

    return np.divide(float(count), float(2*n))


def to_feature_vec(row):
    """
    Returns the feature-vector representation of a piece of input data.
    """

    poly = preprocessing.PolynomialFeatures(degree=2)
    A, B, C, D, E, F, G, H, I = row[0:9]
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
    scorefun = metrics.make_scorer(scorer)
    print("Random Forest with", num_estimators, "estimators.")

    rfc = ensemble.RandomForestClassifier(n_estimators=num_estimators, criterion='entropy', min_samples_split=2, n_jobs=-1)
    scores = cross_validation.cross_val_score(rfc, X, Y, scoring=scorefun, cv = 5)
    print('Mean: ', np.mean(scores), ' +/- ', np.std(scores))

    # OUTPUT

    # Xval = get_features('project_data/validate.csv')
    # # Ypred = gs.best_estimator_.predict(Xval)
    # Ypred = rfc.predict(Xval)
    # # print(Ypred)
    # np.savetxt('out/validate_y.csv', Ypred, delimiter=",", fmt="%i") # the last parameter converts the floats to ints

    # raw_input('Press any key to exit...')


if __name__ == "__main__":
    main()
