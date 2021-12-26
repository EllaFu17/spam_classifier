# Q2
# Gaussian
import numpy as np
import scipy.io
import math

email_dataset = scipy.io.loadmat('/Users/fuyalun/Documents/EE5907-PR/FU YALUN_A0232841L_CA1/src/spamData.mat')

x_train = np.log(np.array(email_dataset["Xtrain"]) + 0.1) # Import data into a matrix
y_train = np.array(email_dataset["ytrain"])
x_test = np.log(np.array(email_dataset["Xtest"]) + 0.1)
y_test = np.array(email_dataset["ytest"])

# y prior
y_train1 = np.sum( y_train == 1) # number of spam emails
y_train_num = np.size(y_train)
y_prior = y_train1 / y_train_num # ML

# spam or not
nonspam_x_train = list()
for i in range (0, y_train.shape[0]):
    if y_train[i] == 0:
        nonspam_x_train.append(i)
spam_x_train = np.delete(x_train, nonspam_x_train, axis=0) # get the features of spam emails
nonspam_x_train = x_train[ [ nonspam_x_train ] ] # get the non-spam emails
# print(spam_x_train)
# average and variance
N_1 = spam_x_train.shape[ 0 ]  # number of rows
avgs_1 = np.mean(spam_x_train, axis=0) # sum by columns, then get the averages(57,1)
vars_1 = np.mean(np.square(spam_x_train - avgs_1), axis=0) # get the variance (57,1),sigma^2
# print('CLASS1_average:', avgs_1)
# print('CLASS1_varriance:', vars_1)

N_0 = nonspam_x_train.shape[ 0 ]
avgs_0 = np.mean(nonspam_x_train, axis=0) # sum by columns, then get the averages(57,1)
vars_0 = np.mean(np.square(nonspam_x_train - avgs_0), axis=0) # get the variance (57,1),sigma^2

# likelihood
gau_train_1 = 1 / np.sqrt(vars_1 * 2 * math.pi) * np.exp(- 0.5 * np.square(x_train - avgs_1) / vars_1 ) # (3065, 57)
gau_train_0 = 1 / np.sqrt(vars_0 * 2 * math.pi) * np.exp(- 0.5 * np.square(x_train - avgs_0) / vars_0 ) # (3065, 57)
gau_train_1 = np.sum(np.log(gau_train_1), axis=1) # (3065, 1)
gau_train_0 = np.sum(np.log(gau_train_0), axis=1) # (3065, 1)

gau_test_1 = 1 / np.sqrt(vars_1 * 2 * math.pi) * np.exp(- 0.5 * np.square(x_test - avgs_1) / vars_1 ) # (1536, 57)
gau_test_0 = 1 / np.sqrt(vars_0 * 2 * math.pi) * np.exp(- 0.5 * np.square(x_test - avgs_0) / vars_0 ) # (1536, 57)
gau_test_1_log = np.log(gau_test_1)
gau_test_0_log = np.log(gau_test_0)
gau_test_1 = np.sum(gau_test_1_log, axis=1) # (1536, 1)
gau_test_0 = np.sum(gau_test_0_log, axis=1) # (1536, 1)

# gaussian list
p_class1_train = math.log(y_prior) + gau_train_1
p_class1_test = math.log(y_prior) + gau_test_1
p_class0_train = math.log(1 - y_prior) + gau_train_0
p_class0_test = math.log(1 - y_prior) + gau_test_0
print(p_class0_test)

# predict
pre_test_gau = np.where (p_class1_test > p_class0_test, 1, 0)
pre_train_gau = np.where (p_class1_train > p_class0_train, 1, 0)
# print(pre_test_gau.shape)
# print(pre_train_gau.shape)

# error rate
error_test = 0
error_train = 0
for e in range(0, y_test.shape[0]):
    if pre_test_gau[e] == y_test[e]:
        error_test = error_test
    else:
        error_test = error_test + 1

for e0 in range(0, y_train.shape[0]):
    if pre_train_gau[e0] == y_train[e0]:
        error_train = error_train
    else:
        error_train = error_train + 1

print('Gaussian')
print('TEST  error rate:', str(error_test/y_test.shape[0]*100) + '%')
print('TRAIN error rate:', str(error_train/y_train.shape[0]*100) + '%') #str(a*100) + '%'
