import numpy as np
import h5py



def load_dataset():
    '''load dataset from h5 file

    :return: train_X, train_Y, test_X, test_Y, classses
    '''
    train_dataset = h5py.File('datasets/train_catvnoncat.h5','r')
    train_set_x_orig = np.array(train_dataset['train_set_x'][:])  # train set features
    train_set_y_orig = np.array(train_dataset['train_set_y'][:])  # train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5','r')
    test_set_x_orig = np.array(test_dataset['test_set_x'][:])     # test set features
    test_set_y_orig = np.array(test_dataset['test_set_y'][:])     # test set labels

    classes = np.array(test_dataset['list_classes'][:]) # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes



def initialize_with_zeros(dim):
    """
    initialize the parameters of logistic regression

    :param dim: dimension of weight
    :return: weight matrix w & bias b
    """
    w = np.zeros((dim, 1))
    b = 0
    return w, b



def sigmoid(z):
    return 1/(1+np.exp(-z))

def propagate(X, Y, w, b):
    """
    FP and BP

    :param X: train_X
    :param Y: train_Y
    :param w: weight
    :param b: bias
    :return: grads and cost
    """
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X)+b)
    cost = -1/m*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))
    dw = 1/m * np.dot(X, (A-Y).T)
    db = 1/m * np.sum(A-Y)
    grads = {
        'dw':dw,
        'db':db
    }
    cost = np.squeeze(cost)
    return grads, cost

def optimize(w, b, train_X, train_Y, num_iterations, learning_rate, print_cost):
    """
    optimize logistic model using Gradient Descent method

    :param w: weight
    :param b: bias
    :param train_X:
    :param train_Y:
    :param num_iterations:
    :param learning_rate:
    :param print_cost:
    :return:
    """

    costs = []
    for i in range(num_iterations):

        grads, cost = propagate(train_X, train_Y, w, b)

        dw = grads['dw']
        db = grads['db']
        # update params
        w = w - grads['dw']*learning_rate
        b = b - grads['db']*learning_rate

        if i%100==0:
            costs.append(cost)

        if print_cost and i % 100==0:
            print("Cost after iteration %i:%f"%(i, cost))

    params={
        'w':w,
        'b':b
    }
    grads={
        'dw':dw,
        'db':db
    }
    return params, grads,costs

def predict(X, w, b):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0],1)

    A = sigmoid(np.dot(w.T, X)+b)
    for i in range(A.shape[1]):
        if A[0][i]>0.5:
            Y_prediction[0][i]=1
        else:
            Y_prediction[0][i]=0

    return Y_prediction

def model(train_X, train_Y, test_X, test_Y, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    Builds the logistic regression model

    :param train_X: training set represented by a numpy array of shape (num_px*num_px*3, m_train)
    :param train_Y: training labels represented by a numpy array (vector) of shape (1, m_train)
    :param test_X:
    :param test_Y:
    :param num_iterations:
    :param learning_rate:
    :param print_cost:
    :return:
    """
    w, b = initialize_with_zeros(train_X.shape[0])
    parameters, grads, costs = optimize(w, b, train_X, train_Y, num_iterations, learning_rate, print_cost)

    w = parameters['w']
    b = parameters['b']
    Y_prediction_test = predict(test_X, w, b)
    Y_prediction_train = predict(train_X, w, b)

    # print accuracy
    print("train accuracy: {} %".format(100-np.mean(np.abs(Y_prediction_train-train_Y))*100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - test_Y)) * 100))
if __name__=='__main__':
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    m_train = train_set_x_orig.shape[0]   # train samples num
    m_test = test_set_x_orig.shape[0]     # test samples num
    num_px = train_set_x_orig[1]   # width(height) of images

    # flatten vectors  (samples_num, img_height, img_width, channels)->(height*width*channels, samples_num)
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

    # standardize dataset
    train_set_x = train_set_x_flatten/255
    test_set_x = test_set_x_flatten/255

    d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate=0.005, print_cost=True)