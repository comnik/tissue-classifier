import numpy

import theano
import theano.tensor as T


def zero_W(n_in, n_out):
    """
    Returns a weight-matrix of shape (n_in, n_out) initialized with 0.
    """

    return numpy.zeros(
        (n_in, n_out),
        dtype=theano.config.floatX
    )


def tanh_W(rng, n_in, n_out):
    """
    Returns uniformely sampled values from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden)).
    This provides suitable initialization of weights for the tanh activation function.

    rng :: numpy.random.RandomState
    rng := A random number generator used to initialize weights
    """

    return numpy.asarray(
        rng.uniform(
            low = -numpy.sqrt(6. / (n_in + n_out)),
            high = numpy.sqrt(6. / (n_in + n_out)),
            size = (n_in, n_out)
        ),
        dtype=theano.config.floatX
    )


def sigmoid_W(rng, n_in, n_out):
    """
    Results presented in [Xavier10] suggest that you
    should use 4 times larger initial weights for sigmoid compared to tanh.

    Parameters are the same as for tanh_W.
    """

    return tanh_W(rng, n_in, n_out) * 4


def zero_bias(n_out):
    """
    Returns a default bias of all zeros.
    """

    return numpy.zeros((n_out,), dtype=theano.config.floatX)


def negative_log_likelihood(y_pred, y):
    """
    Return the mean of the negative log-likelihood of the prediction under a given target distribution.
    We use the mean instead of the sum so that the learningrate is less dependent on the batch size

    y :: theano.tensor.TensorType
    y := A vector that gives for each example the correct label.
    """

    n = y.shape[0]
    return -T.mean(T.log(y_pred)[T.arange(n), y])


def errors(y_pred, y):
    """
    Return a float representing the number of errors in the minibatch
    over the total number of examples of the minibatch; zero one
    loss over the size of the minibatch

    y :: theano.tensor.TensorType
    y := A vector that gives for each example the correct label.
    """

    # check if y has same dimension of y_pred
    if y.ndim != y_pred.ndim:
        raise TypeError(
            'y should have the same shape as self.y_pred',
            ('y', y.type, 'y_pred', y_pred.type)
        )

    # check if y is of the correct datatype
    if y.dtype.startswith('int'):
        # the T.neq operator returns a vector of 0s and 1s, where 1
        # represents a mistake in prediction
        return T.mean(T.neq(y_pred, y))

    else:
        raise NotImplementedError()


def gradient_update(cost, params, learning_rate):
    # compute the gradients of cost, with respect to each parameter
    gparams = [T.grad(cost, param) for param in params]
    # issue updates to the parameters of the model in the form of (variable, update expression) pairs
    return [(param, param - learning_rate * gparam) for param, gparam in zip(params, gparams)]


def gradient_updates_momentum(cost, params, learning_rate, momentum):
    """
    Compute updates for gradient descent with momentum.

    momentum :: float
    Should be at least 0 (standard gradient descent) and less than 1.
    """

    # Make sure momentum is a sane value
    assert momentum < 1 and momentum >= 0
    # List of update steps for each parameter
    updates = []
    # Just gradient descent on cost
    for param in params:
        # For each parameter, we'll create a param_update shared variable.
        # This variable will keep track of the parameter's update step across iterations.
        # We initialize it to 0
        param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
        # Each parameter is updated by taking a step in the direction of the gradient.
        # However, we also "mix in" the previous step according to the given momentum value.
        # Note that when updating param_update, we are using its old value and also the new gradient step.
        updates.append((param, param - learning_rate*param_update))
        # Note that we don't need to derive backpropagation to compute updates - just use T.grad!
        updates.append((param_update, momentum*param_update + (1. - momentum)*T.grad(cost, param)))

    return updates


class Layer(object):

    def __init__(self, n_in, n_out, compute_w, compute_b):
        """
        Typical hidden layer of a MLP, units are fully-connected.
        Weight matrix W is of shape (n_in, n_out) and the bias vector b is of shape (n_out,).

        Hidden unit activation is given by: dot(input, W) + b

        n_in :: Int - dimensionality of input
        n_out :: Int - number of hidden units
        """

        self.W = theano.shared(value=compute_w(n_in, n_out), name='W', borrow=True)
        self.b = theano.shared(value=compute_b(n_out), name='b', borrow=True)

        # parameters of the model
        self.params = [self.W, self.b]

    def output(self, x):
        return T.dot(x, self.W) + self.b


class Sequential(object):
    """
    A simple, sequential composition of layers.
    """

    def __init__(self):
        """
        Initializes a sequential model from a sequence of computations and layers.
        """

        self.steps = []
        self.layers = []

    def layer(self, l):
        """
        Adds a new layer to the model.
        """

        self.steps.append(l.output)
        self.layers.append(l)

        return self

    def step(self, f):
        """
        Adds a new computation step to the model.
        Most useful for activation functions.
        """

        self.steps.append(f)

        return self

    def build(self):
        """
        Finalizes the model.
        """

        # Parameters of the model = parameters of the layers it is made out of.
        self.params = self.layers[0].params
        self.L1 = abs(self.layers[0].W).sum() # L1 norm for regularization.
        self.L2_sqr = (self.layers[0].W ** 2).sum() # Square of L2 norm for regularization.

        for layer in self.layers[1:]:
            self.params += layer.params
            self.L1 += abs(layer.W).sum()
            self.L2_sqr += (layer.W ** 2).sum()

        print("Initialized sequential model with %i layers." % len(self.layers))
        print("Parameters: ", self.params)

        return self

    def output(self, x):
        """
        Compute the networks output for inputs x.
        x :: theano.tensor.var.TensorVariable
        """

        for step in self.steps:
            x = step(x)

        return x
