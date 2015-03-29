import sys
import time
import csv
import datetime

import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from sklearn import preprocessing, metrics, cross_validation, grid_search, neighbors, ensemble

import mlp


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


def meta_classes(Ytrain):
    """
    Converts two dimensional training data (class, label) into
    one dimensional meta-classes for all possible class-label combinations.
    """

    mapping = dict(zip([(1,0), (2,0), (1,1), (2,1), (3,1), (4,1), (4,2), (5,2), (6,2), (7,2)], range(0, 10)))
    return np.atleast_1d([mapping[(c, l)] for (c, l) in Ytrain])


def back_to_normal(Ymeta):
    """
    Converts the meta-class representation back to normal.
    """

    mapping = dict(zip(range(0, 10), [(1,0), (2,0), (1,1), (2,1), (3,1), (4,1), (4,2), (5,2), (6,2), (7,2)]))
    return np.atleast_2d([mapping[c] for c in Ymeta])


def get_features(inpath):
    """
    Reads our input data from a csv file.
    """

    with open(inpath, 'r') as fin:
        reader = csv.reader(fin, delimiter=',')
        X = [to_feature_vec(row) for row in reader]

    return np.atleast_1d(X)


def knn(num_neighbors=4):
    """
    Trains a k-nearest-neighbors model on the training data.
    """

    return neighbors.KNeighborsClassifier(n_neighbors=num_neighbors, weights='distance', p=1)


def random_forest(num_estimators=100):
    """
    Trains a decision tree ensemble on the training data.
    """

    print("Random Forest with %s estimators." % num_estimators)

    return ensemble.RandomForestClassifier(
        n_estimators = num_estimators,
        criterion = 'entropy',
        min_samples_split = 2,
        n_jobs = -1 # try to run on all cores
    )


def shared_dataset(data_xy, borrow=True):
    """
    Function that loads the dataset.
    """

    data_x, data_y = data_xy
    shared_x = np.asarray(data_x, dtype=theano.config.floatX)
    shared_y = np.asarray(data_y, dtype=np.int32)

    return shared_x, shared_y


def neural_network(training_set, validation_set, learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=20):
    """
    Trains a MLP on the training data.
    """

    Xtrain, Ytrain = shared_dataset(training_set)
    Xval, Yval = shared_dataset(validation_set)

    print('Building model...')

    # introduce symbolic vars
    index = T.lscalar() # index to a [mini]batch
    x = T.matrix('x') # the data come as a vector of float features
    y = T.ivector('y') # the labels come as a vector of integer labels

    rng = np.random.RandomState(1234)
    n_in = Xtrain.shape[1]
    n_hidden = 2*n_in
    n_out = 10 # number of possible class / label combinations

    # A simple linear layer that is piped through a nonlinear activation
    # and fed into a linear regression layer.
    # Finally transform outputs into a probability distribution
    # and find the class with highest probability.
    classifier = mlp.Sequential()
    classifier.layer( mlp.Layer(n_in, n_hidden, lambda n1, n2: mlp.tanh_W(rng, n1, n2), mlp.zero_bias) ) \
              .step ( T.tanh ) \
              .layer( mlp.Layer(n_hidden, n_out, mlp.zero_W, mlp.zero_bias) ) \
              .step ( T.nnet.softmax ) \
              .step ( lambda prob: T.argmax(prob, axis=1) ) \
              .build()

    # how to compute the cost we want to minimize during training
    cost = mlp.errors(classifier.output(x), y)
    # (mlp.negative_log_likelihood(classifier.output(x), y) + L1_reg * classifier.L1 + L2_reg * classifier.L2_sqr)

    train = theano.function([x, y], cost, updates=mlp.gradient_update(cost, classifier.params, learning_rate))
    validate = theano.function([x, y], mlp.errors(classifier.output(x), y))
    predict = theano.function([x], classifier.output(x))

    print('Training model...')

    # Early-stopping parameters
    patience = 10000 # look as this many examples regardless
    patience_increase = 2 # wait this much longer when a new best is found
    improvement_threshold = 0.995 # a relative improvement of this much is considered significant

    # how many minibatches to go through before checking the network
    # on the validation set; in this case we check every epoch
    # validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    for epoch in range(0, n_epochs):
        loss = train(Xtrain, Ytrain)
        validation_loss = validate(Xval, Yval)

        print('\t -> Epoch %i, validation error %f%%' % (epoch, validation_loss * 100.))

        # if we got the best validation score until now
        if validation_loss < best_validation_loss:
            # #improve patience if loss improvement is good enough
            # if (validation_loss < best_validation_loss * improvement_threshold):
            #     patience = max(patience, iter * patience_increase)

            best_validation_loss = validation_loss
            best_iter = epoch

    end_time = time.clock()
    print('Optimization complete. Best validation score of %f %% obtained at iteration %i.' %
          (best_validation_loss * 100., best_iter + 1))

    print >> sys.stderr, ('Time taken: %.2fm' % ((end_time - start_time) / 60.))


def main():
    plt.ion()

    # Read labelled training data.
    X = get_features('project_data/train.csv')
    Y = meta_classes(np.genfromtxt('project_data/train_y.csv', delimiter=','))

    Xtrain, Xtest, Ytrain, Ytest = cross_validation.train_test_split(X, Y, train_size=0.75)
    training_set = (Xtrain, Ytrain)
    validation_set = (Xtest, Ytest)

    # classifier = random_forest()
    # scorefun = metrics.make_scorer(scorer)
    # scores = cross_validation.cross_val_score(classifier, X, Y, scoring=scorefun, cv=5)
    # print('Mean: %s +/- %s' % (np.mean(scores), np.std(scores)))

    neural_network(training_set, validation_set)

    # OUTPUT
    # Xval = get_features('project_data/validate.csv')
    # Ypred = classifier.predict(Xval)
    # np.savetxt('out/validate_y.csv', Ypred, delimiter=",", fmt="%i") # the last parameter converts the floats to ints

    # raw_input('Press any key to exit...')


if __name__ == "__main__":
    main()
