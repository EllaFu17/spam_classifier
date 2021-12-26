# Q4
# KNN
import numpy as np
import heapq
import matplotlib.pyplot as plt
import scipy.io

def eu_dist(a, b, n):
    dist = []
    for j in range (0, a.shape[0]):
          dist_j = np.sqrt(np.sum(np.square(b - a[j]), axis=1)) #(3065,)
          dist_j = list(dist_j)
          # find the index of the k samples with the smallest distance
          dist_j = map(dist_j.index, heapq.nsmallest(n, dist_j))
          dist_j = list(dist_j) # k
          dist.append(dist_j) #3065,k
    dist = np.array(dist)
    return dist

def predict_y(y, index, m):
    n = index.shape[0]
    index = index.reshape((m*n)).astype('int32')
    index_list = index.tolist()
    # print('indexshape', index.shape)
    # print(len(index_list))
    lab = y [index_list]
    lab = lab.reshape(n,m) # (3065,k)
    # print('index', lab)
    pre = np.sum(lab, axis=1)  # sum by row
    # print('preshape', pre.shape)
    for i in range (0, pre.shape[0]):
        if pre[i] >= m/2:
            pre[i] = 1
        else:
            pre[i] = 0
    # pre = pre.reshape((y.shape[0], ind.shape[1]))
    return pre  # (3065,)

if __name__ == '__main__':
    email_dataset = scipy.io.loadmat('spamData.mat')
    x_train = np.log(np.array(email_dataset[ "Xtrain" ]) + 0.1)  # Import data into a matrix
    y_train = np.array(email_dataset[ "ytrain" ])
    x_test = np.log(np.array(email_dataset[ "Xtest" ]) + 0.1)
    y_test = np.array(email_dataset[ "ytest" ])
    test_error_rate_list = list()
    train_error_rate_list = list()
    K1 = np.linspace(start=1, stop=9, num=9, dtype=int)
    K2 = np.linspace(start=10, stop=100, num=19, dtype=int)
    K_list = np.append(K1, K2)
    for k in K_list:
        eu_train = eu_dist(x_train, x_train, k)  # (3065,k)
        eu_test = eu_dist(x_test, x_train, k)
        # print('eu_test', eu_test.shape)
        pre_train = predict_y(y_train, eu_train, k)  # (3065,k)
        pre_test = predict_y(y_train, eu_test, k)
        # print('pre_train', pre_train.shape)
        # print('pre_test', pre_test.shape)
        error_train = 0
        error_test = 0
        for e in range(0, y_train.shape[ 0 ]):
            if pre_train[ e ] == y_train[ e ]:
                error_train = error_train
            else:
                error_train = error_train + 1
        train_error_rate = 100 * error_train / y_train.shape[ 0 ]
        train_error_rate_list.append(train_error_rate)
        train_error_rate = str(train_error_rate) + '%'

        for e0 in range(0, y_test.shape[ 0 ]):
            if pre_test[ e0 ] == y_test[ e0 ]:
                error_test = error_test
            else:
                error_test = error_test + 1
        test_error_rate = 100 * error_test / y_test.shape[ 0 ]
        test_error_rate_list.append(test_error_rate)
        test_error_rate = str(test_error_rate) + '%'
        if k == 1:
          print('K =', k)
          print('TRAIN error rate:', train_error_rate)  # str(a*100) + '%'
          print('TEST error rate:', test_error_rate)

        elif k == 10:
          print('K =', k)
          print('TRAIN error rate:', train_error_rate)  # str(a*100) + '%'
          print('TEST error rate:', test_error_rate)

        elif k == 100:
          print('K =', k)
          print('TRAIN error rate:', train_error_rate)  # str(a*100) + '%'
          print('TEST error rate:', test_error_rate)

    plt.title('KNN')
    # plt.ylim((11, 15))
    plt.xlabel('K')
    plt.ylabel('Error Rate %')
    plt.plot(K_list, test_error_rate_list, color = 'blue', label='Test')
    plt.plot(K_list, train_error_rate_list, color = 'red', label='Train')
    plt.legend(loc='upper left')
    plt.show()
