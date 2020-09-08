import numpy as np
import os
import tools
import matplotlib.pyplot as plt

########################

# 33_frrn_slim_valid_scheduled
# 34_frrn_slim_valid_deep_scheduled
# 35_frrn_slim_valid_2_scheduled

sm = 3

names = 'C1O_16_L_NNSC','C1O_16_L_NNSC_WD'

smooth_window = 0
n_train_oas = []
n_train_mean_f1s = []
n_valid_oas = []
n_valid_mean_f1s = []
n_test_oas = []
n_test_mean_f1s = []

for name in names:
    root = './results/' + name + '/confusion_matrices/'
    n = max([int(txt.split('_')[-1].split('.')[0]) for txt in os.listdir(root)]) + 1

    train_oas = np.zeros(n)
    train_mean_f1s = np.zeros(n)
    valid_oas = np.zeros(n)
    valid_mean_f1s = np.zeros(n)
    test_oas = np.zeros(n)
    test_mean_f1s = np.zeros(n)

    for i in range(n):
        confusions = np.load(root + 'CM_TRAINING_{}.npy'.format(i))
        pctgs, precisions, recall, f1, mean_f1, oa = tools.get_confusion_metrics(confusions)
        train_oas[i] = oa
        train_mean_f1s[i] = mean_f1

        confusions = np.load(root + 'CM_VALIDATION_{}.npy'.format(i))
        pctgs, precisions, recall, f1, mean_f1, oa = tools.get_confusion_metrics(confusions)
        valid_oas[i] = oa
        valid_mean_f1s[i] = mean_f1

        try:
            confusions = np.load(root + 'CM_TESTING_{}.npy'.format(i))
            pctgs, precisions, recall, f1, mean_f1, oa = tools.get_confusion_metrics(confusions)
            test_oas[i] = oa
            test_mean_f1s[i] = mean_f1
        except IOError:
            test_oas[i] = test_oas[i - 1]
            test_mean_f1s[i] = test_mean_f1s[i - 1]

    if smooth_window:
        train_oas = tools.smooth1d(train_oas, smooth_window)
        train_mean_f1s = tools.smooth1d(train_mean_f1s, smooth_window)
        valid_oas = tools.smooth1d(valid_oas, smooth_window)
        valid_mean_f1s = tools.smooth1d(valid_mean_f1s, smooth_window)
        test_oas = tools.smooth1d(test_oas, smooth_window)
        test_mean_f1s = tools.smooth1d(test_mean_f1s, smooth_window)

    n_train_oas.append(train_oas)
    n_train_mean_f1s.append(train_mean_f1s)
    n_valid_oas.append(valid_oas)
    n_valid_mean_f1s.append(valid_mean_f1s)
    n_test_oas.append(test_oas)
    n_test_mean_f1s.append(test_mean_f1s)

    # best_train_oa_epo = np.argmax(train_oas)
    # best_train_oa_val = train_oas[best_train_oa_epo]*100
    # best_valid_oa_epo = np.argmax(valid_oas)
    # best_valid_oa_val = valid_oas[best_valid_oa_epo]*100
    # best_test_oa_epo = np.argmax(test_oas)
    # best_test_oa_val = test_oas[best_test_oa_epo]*100
    #
    # best_train_f1_epo = np.argmax(train_mean_f1s)
    # best_train_f1_val = train_mean_f1s[best_train_f1_epo]*100
    # best_valid_f1_epo = np.argmax(valid_mean_f1s)
    # best_valid_f1_val = valid_mean_f1s[best_valid_f1_epo]*100
    # best_test_f1_epo = np.argmax(test_mean_f1s)
    # best_test_f1_val = test_mean_f1s[best_test_f1_epo]*100

N = ''
for n in names:
    N += n
    N += ' vs. '

plt.suptitle(N[:-4])

linestyles = ['-', '--', '-.']
colors = ['red', 'green', 'blue', 'black', 'cyan', 'orange', 'magenta']

plt.subplot(1, 2, 1)
plt.title('Overall accuracies')
for i in range(len(names)):
    plt.plot(tools.smooth1d(np.array(n_train_oas[i]), sm) * 100, label='Training - ' + names[i], linestyle=linestyles[0],
             color=colors[i])
    plt.plot(tools.smooth1d(np.array(n_valid_oas[i]), sm) * 100, label='Validation - ' + names[i], linestyle=linestyles[1],
             color=colors[i])
    plt.plot(tools.smooth1d(np.array(n_test_oas[i]), sm) * 100, label='Testing - ' + names[i], linestyle=linestyles[2],
             color=colors[i])
plt.ylim((0, 100))
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Overall accuracy [%]')

# plt.hlines(y=best_train_oa_val, xmin=0, xmax=n, linewidth=1, color='blue')
# plt.vlines(x=best_train_oa_epo, ymin=0, ymax=100, linewidth=1, color='blue')
# plt.hlines(y=best_valid_oa_val, xmin=0, xmax=n, linewidth=1, color='orange')
# plt.vlines(x=best_valid_oa_epo, ymin=0, ymax=100, linewidth=1, color='orange')
# plt.hlines(y=best_test_oa_val, xmin=0, xmax=n, linewidth=1, color='green')
# plt.vlines(x=best_test_oa_epo, ymin=0, ymax=100, linewidth=1, color='green')
# # plt.savefig('run1000.png', bbox_inches='tight')

# MEAN F1 SCORE
plt.subplot(1, 2, 2)
plt.title('Mean F1 scores')
for i in range(len(names)):
    plt.plot(tools.smooth1d(np.array(n_train_mean_f1s[i]), sm) * 100, label='Training - ' + names[i][:2], linestyle=linestyles[0],
             color=colors[i])
    plt.plot(tools.smooth1d(np.array(n_valid_mean_f1s[i]), sm) * 100, label='Validation - ' + names[i][:2], linestyle=linestyles[1],
             color=colors[i])
    plt.plot(tools.smooth1d(np.array(n_test_mean_f1s[i]), sm) * 100, label='Testing - ' + names[i][:2], linestyle=linestyles[2],
             color=colors[i])
plt.ylim((0, 100))
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Mean F1 score [%]')

# plt.hlines(y=best_train_f1_val, xmin=0, xmax=n, linewidth=1, color='blue')
# plt.vlines(x=best_train_f1_epo, ymin=0, ymax=100, linewidth=1, color='blue')
# plt.hlines(y=best_valid_f1_val, xmin=0, xmax=n, linewidth=1, color='orange')
# plt.vlines(x=best_valid_f1_epo, ymin=0, ymax=100, linewidth=1, color='orange')
# plt.hlines(y=best_test_f1_val, xmin=0, xmax=n, linewidth=1, color='green')
# plt.vlines(x=best_test_f1_epo, ymin=0, ymax=100, linewidth=1, color='green')
# # plt.savefig('run1000.png', bbox_inches='tight')
plt.show()
