# Q3
# Logistic Regression
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

def l_regression(u, w_reg):
    times = 0
    for times in range(0, 15):
        g = np.dot(np.transpose(x_train_reg), u - y_train)  # (58, 1)
        u = u.reshape(-1)
        S = np.diag(u * (1 - u))
        H = x_train_reg.T.dot(S).dot(x_train_reg)
        w_lam = w_reg.copy()
        w_lam[ 0, 0 ] = 0
        g_reg = g + lam * w_lam
        H_reg = H + lam * I_reg  #
        try:
            H_n = np.linalg.inv(H_reg)
        except np.linalg.LinAlgError:  # 奇异矩阵
            H_n = H_reg
        descent = np.dot(H_n, g_reg)
        w_new = w_reg - descent
        if (np.abs(w_new - w_reg)).all() <= 0.0001:
            # print('END')
            break
        w_reg = w_new
        z = np.dot(x_train_reg, w_reg)
        u = 1 / (1 + np.exp(-z))

    y_train_predict = u
    y_test_predict = 1 / (1 + np.exp(-np.dot(x_test_reg, w_reg)))
    pre_train = np.where(y_train_predict >= 0.5, 1, 0)
    pre_test = np.where(y_test_predict >= 0.5, 1, 0)
    error_train = 0
    error_test = 0
    for e in range(0, y_train.shape[ 0 ]):
        if pre_train[ e ] == y_train[ e ]:
            error_train = error_train
        else:
            error_train = error_train + 1
    train_error_rate = 100 * error_train / 3065
    train_error_rate_list.append(train_error_rate)
    train_error_rate = str(train_error_rate) + '%'

    for e in range(0, y_test.shape[ 0 ]):
        if pre_test[ e ] == y_test[ e ]:
            error_test = error_test
        else:
            error_test = error_test + 1
    test_error_rate = 100 * error_test / 1536
    test_error_rate_list.append(test_error_rate)
    test_error_rate = str(test_error_rate) + '%'

    if lam == 1:
      print('lambda =', lam)
      # print('times:', times)
      print('TRAIN error rate:', train_error_rate)  # str(a*100) + '%'
      print('TEST error rate:', test_error_rate)

    elif lam == 10:
      print('lambda =', lam)
      print('TRAIN error rate:', train_error_rate)  # str(a*100) + '%'
      print('TEST error rate:', test_error_rate)

    elif lam == 100:
      print('lambda =', lam)
      print('TRAIN error rate:', train_error_rate)  # str(a*100) + '%'
      print('TEST error rate:', test_error_rate)


if __name__ == '__main__':
    email_dataset = scipy.io.loadmat('spamData.mat')
    x_train = np.log(np.array(email_dataset[ "Xtrain" ]) + 0.1)  # Import data into a matrix
    y_train = np.array(email_dataset[ "ytrain" ])
    x_test = np.log(np.array(email_dataset[ "Xtest" ]) + 0.1)
    y_test = np.array(email_dataset[ "ytest" ])

    # lam = λ = {1, 2, · · · , 9, 10, 15, 20, · · · , 95, 100}
    lam1 = np.linspace(start=1, stop=9, num=9)
    lam2 = np.linspace(start=10, stop=100, num=19)
    lam_list = np.append(lam1, lam2)
    test_error_rate_list = list()
    train_error_rate_list = list()
    x_train_reg = np.concatenate((np.ones((x_train.shape[ 0 ], 1)), x_train), axis=1)  # (3065, 58)
    x_test_reg = np.concatenate((np.ones((x_test.shape[ 0 ], 1)), x_test), axis=1)  # (3065, 58)
    # z = np.dot(x_train_reg, w_reg)  # (58,1)
    # u = 1 / (1 + np.exp(-z))
    I_reg = np.identity(58)  # (58, 58)
    I_reg[ 0, 0 ] = 0
    for lam in lam_list:
        w = np.zeros((58, 1))  # (58,1)
        sigmoid = 1 / (1 + np.exp(-np.dot(x_train_reg, np.zeros((58, 1)))))
        l_regression(sigmoid, w)
    plt.title('Logistic Regression')
    plt.xlabel('lambda (λ)')
    plt.ylabel('Error Rate %')
    plt.ylim((4.5, 7))
    plt.plot(lam_list, test_error_rate_list, color = 'blue', label='Test')
    plt.plot(lam_list, train_error_rate_list, color = 'red', label='Train')
    plt.legend(loc='upper left')
    plt.show()
