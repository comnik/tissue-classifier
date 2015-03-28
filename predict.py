import sys
import time
import csv
import datetime

import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from sklearn import preprocessing, metrics, cross_validation, grid_search, neighbors, ensemble
from mlp     import MLP


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
    Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """

    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)

    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')


def mlp(training_set, validation_set,
        learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000, batch_size=20, n_hidden=250):
    """
    Trains a MLP on the training data.
    """

    Xtrain, Ytrain = shared_dataset(training_set)
    Xval, Yval = shared_dataset(validation_set)

    # compute number of minibatches for training, validation and testing
    n_train_batches = Xtrain.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = Xval.get_value(borrow=True).shape[0] / batch_size

    print('Building model...')

    # introduce symbolic vars
    index = T.lscalar() # index to a [mini]batch
    x = T.matrix('x') # the data come as a vector of float features
    y = T.ivector('y') # the labels come as a vector of integer labels

    rng = np.random.RandomState(1234)

    classifier = MLP(
        rng = rng,
        input = x,
        n_in = Xtrain.get_value(borrow=True).shape[1],
        n_hidden = n_hidden,
        n_out = 10 # number of possible class / label combinations
    )

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    validate_model = theano.function(
        inputs = [index],
        outputs = classifier.errors(y),
        givens = {
            x: Xval[index * batch_size:(index + 1) * batch_size],
            y: Yval[index * batch_size:(index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta (sorted in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs
    updates = [(param, param - learning_rate * gparam) for param, gparam in zip(classifier.params, gparams)]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules defined in `updates`
    train_model = theano.function(
        inputs = [index],
        outputs = cost,
        updates = updates,
        givens = {
            x: Xtrain[index * batch_size:(index + 1) * batch_size],
            y: Ytrain[index * batch_size:(index + 1) * batch_size]
        }
    )

    print('Training model...')

    # Early-stopping parameters
    patience = 10000 # look as this many examples regardless
    patience_increase = 2 # wait this much longer when a new best is found
    improvement_threshold = 0.995 # a relative improvement of this much is considered significant

    # how many minibatches to go through before checking the network
    # on the validation set; in this case we check every epoch
    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    for epoch in range(0, n_epochs):
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index # iteration number

            if (iter + 1) % validation_frequency == 0:
                validation_losses = map(validate_model, xrange(n_valid_batches)) # zero-one loss on validation set
                this_validation_loss = np.mean(validation_losses)

                print('\t -> Epoch %i, minibatch %i/%i, validation error %f %%' % (
                    epoch,
                    minibatch_index + 1,
                    n_train_batches,
                    this_validation_loss * 100.
                ))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (this_validation_loss < best_validation_loss * improvement_threshold):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

            if patience <= iter:
                break

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

    mlp(training_set, validation_set)

    # OUTPUT
    # Xval = get_features('project_data/validate.csv')
    # Ypred = classifier.predict(Xval)
    # np.savetxt('out/validate_y.csv', Ypred, delimiter=",", fmt="%i") # the last parameter converts the floats to ints

    # raw_input('Press any key to exit...')


if __name__ == "__main__":
    main()
