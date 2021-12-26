# Q1
# beta
import numpy as np
import scipy.io
import math
import matplotlib.pyplot as plt

def class1(a):
    p_list = list()
    a = a / 2
    for m in range(0, 57):
        p = (N1_1[ m ] + a) / (N_1 + a + a)  # beta a = b
        p_list.append(p)
    p_list = np.array(p_list)
    p_list0 = np.log(1 - p_list + 1e-10)
    p_list = np.log(p_list)
    # print(p_list0.shape)
    class1_test = math.log(y_prior) + np.dot(x_test, p_list) + np.dot(np.logical_not(x_test).astype(int),p_list0)  # 取反
    class1_train = math.log(y_prior) + np.dot(x_train, p_list) + np.dot(np.logical_not(x_train).astype(int), p_list0)
    # print(class1_test.shape) # 1536
    return class1_test, class1_train

def class0(a):
    p0_list = []
    a = a/2
    for m0 in range(0, 57):
        p = (N1_0[ m0 ] + a) / (N_0 + a + a)  # beta a = b
        p0_list.append(p)
    p0_list = np.array(p0_list)
    p0_list0 = np.log(1 - p0_list + 1e-10)
    p0_list = np.log(p0_list)
    # print(p0_list.shape)
    class0_test = math.log(1 - y_prior) + np.dot(x_test, p0_list) + np.dot(np.logical_not(x_test).astype(int), p0_list0)  # 取反
    class0_train = math.log(1 - y_prior) + np.dot(x_train, p0_list) + np.dot(np.logical_not(x_train).astype(int), p0_list0)
    return class0_test, class0_train

    # print(p_class0_test.shape)


if __name__ == '__main__':
    # Import data into a matrix
    email_dataset = scipy.io.loadmat('spam/src/spamData.mat')
    x_train00 = np.array(email_dataset[ "Xtrain" ])
    y_train = np.array(email_dataset[ "ytrain" ])
    x_test00 = np.array(email_dataset[ "Xtest" ])
    y_test = np.array(email_dataset[ "ytest" ])
    # convert the x_train and x_test into 0 and 1
    x_train = np.where(x_train00 > 0, 1, 0)
    x_test = np.where(x_test00 > 0, 1, 0)
    # y prior
    y_train1 = np.sum(y_train == 1)  # number of spam
    y_train_num = np.size(y_train)
    y_prior = y_train1 / y_train_num  # ML
    # spam or not
    spam_num0 = list()
    for i in range(0, y_train.shape[0]):
        if y_train[ i ] == 0:
            spam_num0.append(i)
    spam_num1 = np.delete(x_train, spam_num0, axis=0)  # get the features of spam emails
    # class1
    N_1 = spam_num1.shape[ 0 ]  # number of rows
    N1_1 = np.sum(spam_num1, axis=0)  # sum by column

    # class0
    N_0 = len(spam_num0)
    non_spam = x_train[ [ spam_num0 ] ] # get the features of non-spam emails
    N1_0 = np.sum(non_spam, axis=0)
    # beta
    p_class1_test = []
    p_class1_train = []
    p_class0_test = []
    p_class0_train = []
    test_error_rate_list = list()
    train_error_rate_list = list()
    beta_all = np.linspace(start = 0, stop = 200, num = 201)
    for i in range (0, 201):
        # beta_a = beta_all[i]*2
        beta_a = beta_all[i]
        beta_a = int (beta_a)
        p_class1_test, p_class1_train = class1(beta_a)
        p_class0_test, p_class0_train = class0(beta_a)
        predict_test = np.where(p_class1_test > p_class0_test, 1, 0)
        # print(predict_test)
        # np.savetxt('predict_test',predict_test)
        predict_train = np.where(p_class1_train > p_class0_train, 1, 0)
        # print(predict_train)

        # error rate
        error_test = 0
        error_train = 0
        for e in range(0, 1536):
            if predict_test[ e ] == y_test[ e ]:
                error_test = error_test
            else:
                error_test = error_test + 1
        test_error_rate = 100 * error_test / 1536
        test_error_rate_list.append(test_error_rate)
        test_error_rate = str(test_error_rate) + '%'

        for e0 in range(0, 3065):
            if predict_train[ e0 ] == y_train[ e0 ]:
                error_train = error_train
            else:
                error_train = error_train + 1
        train_error_rate = 100 * error_train / 3065
        train_error_rate_list.append(train_error_rate)
        train_error_rate = str(train_error_rate) + '%'

        if beta_a/2 == 1:
            print('BETA a = b =', beta_a/2)
            print('TEST  error rate:', test_error_rate)
            print('TRAIN error rate:', train_error_rate)  # str(a*100) + '%'
        elif beta_a/2 == 10:
            print('BETA a = b =', beta_a/2)
            print('TEST  error rate:', test_error_rate)
            print('TRAIN error rate:', train_error_rate)  # str(a*100) + '%'
        elif beta_a/2 == 100:
            print('BETA a = b =', beta_a/2)
            print('TEST  error rate:', test_error_rate)
            print('TRAIN error rate:', train_error_rate)  # str(a*100) + '%'

    plt.title('Error Rates_Beta Distribution')
    plt.ylim((11, 15))
    plt.xlabel('Beta(a,b)')
    plt.ylabel('Error Rate %')
    plt.plot(0.5 * beta_all, test_error_rate_list, color = 'blue', label='Test')
    plt.plot(0.5 * beta_all, train_error_rate_list, color = 'red', label='Train')
    plt.legend(loc='upper left')
    plt.show()
    # plt.close()
